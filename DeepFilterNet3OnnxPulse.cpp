#include "DeepFilterNet3OnnxPulse.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

#ifdef __ANDROID__
#include <android/log.h>
#define DFN_PULSE_LOG(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, "OnnxPulse", fmt, ##__VA_ARGS__)
#else
#define DFN_PULSE_LOG(fmt, ...) fprintf(stderr, "[OnnxPulse] " fmt "\n", ##__VA_ARGS__)
#endif

namespace {

size_t numelFromShape(const std::vector<int64_t> &shape) {
    size_t n = 1;
    for (int64_t d : shape) {
        if (d <= 0) continue;
        n *= static_cast<size_t>(d);
    }
    return n;
}

} // namespace

std::string DeepFilterNet3OnnxPulse::guessStateOutputForInput(const std::string &in,
                                                              const std::vector<std::string> &outputs) {
    const std::vector<std::string> candidates = {
        std::string("out_") + in,
        in + "_out",
        std::string("o_") + in,
        std::string("out_state_") + in,
    };
    for (const auto &c : candidates) {
        if (std::find(outputs.begin(), outputs.end(), c) != outputs.end())
            return c;
    }
    if (in.size() > 6 && in.compare(0, 6, "state_") == 0) {
        const std::string tail = in.substr(6);
        const std::string o = std::string("out_state_") + tail;
        if (std::find(outputs.begin(), outputs.end(), o) != outputs.end())
            return o;
    }
    if (std::find(outputs.begin(), outputs.end(), in) != outputs.end())
        return in;
    return {};
}

int DeepFilterNet3OnnxPulse::indexOfName(const std::vector<std::string> &names, const std::string &n) {
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == n)
            return static_cast<int>(i);
    }
    return -1;
}

std::vector<int64_t> DeepFilterNet3OnnxPulse::tensorShape(const Ort::Value &v) {
    return v.GetTensorTypeAndShapeInfo().GetShape();
}

size_t DeepFilterNet3OnnxPulse::shapeNumel(const std::vector<int64_t> &shape) {
    return numelFromShape(shape);
}

DeepFilterNet3OnnxPulse::Subgraph DeepFilterNet3OnnxPulse::loadSubgraph(
    Ort::Env &env, Ort::SessionOptions &opt, const char *path,
    std::unordered_set<std::string> feature_names) {
    Subgraph g;
    g.feature_names = std::move(feature_names);
    g.session = std::make_unique<Ort::Session>(env, path, opt);

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t n_in = g.session->GetInputCount();
    const size_t n_out = g.session->GetOutputCount();

    g.input_names.reserve(n_in);
    for (size_t i = 0; i < n_in; ++i) {
        auto name_alloc = g.session->GetInputNameAllocated(i, allocator);
        g.input_names.emplace_back(name_alloc.get());
    }
    g.output_names.reserve(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        auto name_alloc = g.session->GetOutputNameAllocated(i, allocator);
        g.output_names.emplace_back(name_alloc.get());
    }

    for (size_t ii = 0; ii < g.input_names.size(); ++ii) {
        const std::string &in_name = g.input_names[ii];
        if (g.feature_names.count(in_name))
            continue;
        Ort::TypeInfo type_info = g.session->GetInputTypeInfo(ii);
        if (type_info.GetONNXType() != ONNX_TYPE_TENSOR)
            throw std::runtime_error("OnnxPulse: non-tensor state input: " + in_name);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("OnnxPulse: non-float32 state input not supported: " + in_name);
        std::vector<int64_t> shape = tensor_info.GetShape();
        for (int64_t &d : shape) {
            if (d < 0)
                d = 1;
        }
        const size_t n = numelFromShape(shape);
        g.state_shapes[in_name] = shape;
        g.state_buffers[in_name].assign(n, 0.f);
        // libDF：atten_lim_db>=100 表示不限制、不在谱上混回噪声（tract.rs）；LL ONNX 常沿用 100 作「无限制」标量
        if (in_name == "atten_lim" || in_name.find("atten_lim") != std::string::npos)
            std::fill(g.state_buffers[in_name].begin(), g.state_buffers[in_name].end(), 100.f);
    }

    for (const auto &kv : g.state_buffers) {
        const std::string &state_in = kv.first;
        std::string out_name = guessStateOutputForInput(state_in, g.output_names);
        g.state_in_to_out[state_in] = std::move(out_name);
    }

    return g;
}

void DeepFilterNet3OnnxPulse::resetSubgraph(Subgraph &g) {
    for (auto &kv : g.state_buffers) {
        if (kv.first == "atten_lim" || kv.first.find("atten_lim") != std::string::npos)
            std::fill(kv.second.begin(), kv.second.end(), 100.f);
        else
            std::fill(kv.second.begin(), kv.second.end(), 0.f);
    }
}

std::vector<Ort::Value> DeepFilterNet3OnnxPulse::buildInputs(
    Subgraph &g, Ort::MemoryInfo &mem,
    const std::unordered_map<std::string, std::pair<const float *, std::vector<int64_t>>> &features) {
    std::vector<Ort::Value> inputs;
    inputs.reserve(g.input_names.size());

    for (const std::string &in_name : g.input_names) {
        auto feat_it = features.find(in_name);
        if (feat_it != features.end()) {
            const float *data = feat_it->second.first;
            const std::vector<int64_t> &shape = feat_it->second.second;
            const size_t n = numelFromShape(shape);
            inputs.push_back(Ort::Value::CreateTensor<float>(
                mem, const_cast<float *>(data), n, shape.data(), shape.size()));
        } else {
            auto st = g.state_buffers.find(in_name);
            auto sh = g.state_shapes.find(in_name);
            if (st == g.state_buffers.end() || sh == g.state_shapes.end()) {
                inputs.clear();
                return inputs;
            }
            std::vector<float> &buf = st->second;
            const std::vector<int64_t> &shape = sh->second;
            inputs.push_back(Ort::Value::CreateTensor<float>(
                mem, buf.data(), buf.size(), shape.data(), shape.size()));
        }
    }

    return inputs;
}

void DeepFilterNet3OnnxPulse::updateStateFromOutputs(Subgraph &g, std::vector<Ort::Value> &outs) {
    if (outs.size() != g.output_names.size())
        return;

    std::unordered_map<std::string, const float *> out_ptr;
    out_ptr.reserve(outs.size());
    for (size_t i = 0; i < outs.size(); ++i) {
        out_ptr[g.output_names[i]] = outs[i].GetTensorMutableData<float>();
    }

    for (const auto &kv : g.state_in_to_out) {
        const std::string &state_in = kv.first;
        const std::string &out_name = kv.second;
        if (out_name.empty())
            continue;
        auto op = out_ptr.find(out_name);
        auto bu = g.state_buffers.find(state_in);
        if (op == out_ptr.end() || bu == g.state_buffers.end())
            continue;
        const float *src = op->second;
        std::vector<float> &dst = bu->second;
        if (dst.empty())
            continue;
        std::memcpy(dst.data(), src, dst.size() * sizeof(float));
    }
}

size_t DeepFilterNet3OnnxPulse::queryOutputFloatNumel(const Subgraph &g, const char *output_name) {
    const int i = indexOfName(g.output_names, output_name);
    if (i < 0)
        return 0;
    Ort::TypeInfo t = g.session->GetOutputTypeInfo(static_cast<size_t>(i));
    if (t.GetONNXType() != ONNX_TYPE_TENSOR)
        return 0;
    std::vector<int64_t> shape = t.GetTensorTypeAndShapeInfo().GetShape();
    for (int64_t &d : shape) {
        if (d < 0)
            d = 1;
    }
    return numelFromShape(shape);
}

DeepFilterNet3OnnxPulse::DeepFilterNet3OnnxPulse(Ort::Env &env, Ort::SessionOptions &session_options,
                                                 const Config &cfg)
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    if (cfg.nb_erb < 1 || cfg.nb_df < 1 || cfg.df_order < 0)
        throw std::runtime_error("DeepFilterNet3OnnxPulse: invalid nb_erb/nb_df/df_order");
    nb_erb_ = cfg.nb_erb;
    nb_df_ = cfg.nb_df;
    df_order_ = cfg.df_order;

    enc_ = loadSubgraph(env, session_options, cfg.enc_path,
                        {"feat_erb", "feat_spec"});
    erb_ = loadSubgraph(env, session_options, cfg.erb_path,
                        {"emb", "e3", "e2", "e1", "e0"});
    df_ = loadSubgraph(env, session_options, cfg.df_path,
                       {"emb", "c0"});

    const size_t meta_m = queryOutputFloatNumel(erb_, "m");
    const size_t meta_c = queryOutputFloatNumel(df_, "coefs");
    const size_t need_m = static_cast<size_t>(nb_erb_);
    const size_t need_c =
        static_cast<size_t>(nb_df_) * static_cast<size_t>(df_order_) * 2u;
    // 元数据在动态维下会严重低估 numel；缓冲区至少按 config [df] 分配
    erb_m_numel_ = std::max(meta_m, need_m);
    df_coefs_numel_ = std::max(meta_c, need_c);
}

void DeepFilterNet3OnnxPulse::reset() {
    resetSubgraph(enc_);
    resetSubgraph(erb_);
    resetSubgraph(df_);
}

bool DeepFilterNet3OnnxPulse::runFrame(const float *feat_erb, const float *feat_spec,
                                       float *out_m, size_t out_m_elems,
                                       float *out_coefs, size_t out_coefs_elems) {
    if (!feat_erb || !feat_spec || !out_m || !out_coefs) {
        DFN_PULSE_LOG("runFrame: null pointer");
        return false;
    }

    const std::array<int64_t, 4> erb_shape{1, 1, 1, static_cast<int64_t>(nb_erb_)};
    const std::array<int64_t, 4> spec_shape{1, 2, 1, static_cast<int64_t>(nb_df_)};

    std::unordered_map<std::string, std::pair<const float *, std::vector<int64_t>>> enc_feat;
    enc_feat["feat_erb"] = {feat_erb, std::vector<int64_t>(erb_shape.begin(), erb_shape.end())};
    enc_feat["feat_spec"] = {feat_spec, std::vector<int64_t>(spec_shape.begin(), spec_shape.end())};

    std::vector<Ort::Value> enc_inputs = buildInputs(enc_, memory_info_, enc_feat);
    if (enc_inputs.size() != enc_.input_names.size()) {
        DFN_PULSE_LOG("enc buildInputs: got %zu tensors, need %zu (check feat/state names)",
                      enc_inputs.size(), enc_.input_names.size());
        return false;
    }

    std::vector<const char *> enc_in_ptrs;
    std::vector<const char *> enc_out_ptrs;
    enc_in_ptrs.reserve(enc_.input_names.size());
    enc_out_ptrs.reserve(enc_.output_names.size());
    for (const auto &s : enc_.input_names)
        enc_in_ptrs.push_back(s.c_str());
    for (const auto &s : enc_.output_names)
        enc_out_ptrs.push_back(s.c_str());

    std::vector<Ort::Value> enc_outs;
    try {
        enc_outs = enc_.session->Run(Ort::RunOptions{nullptr},
                                     enc_in_ptrs.data(), enc_inputs.data(), enc_inputs.size(),
                                     enc_out_ptrs.data(), enc_out_ptrs.size());
    } catch (const Ort::Exception &e) {
        DFN_PULSE_LOG("enc Run: %s", e.what());
        return false;
    }

    updateStateFromOutputs(enc_, enc_outs);

    std::unordered_map<std::string, std::pair<const float *, std::vector<int64_t>>> erb_feat;
    for (const std::string &name : erb_.feature_names) {
        const int ei = indexOfName(enc_.output_names, name);
        if (ei < 0) {
            DFN_PULSE_LOG("enc output missing name '%s'", name.c_str());
            return false;
        }
        float *p = enc_outs[static_cast<size_t>(ei)].GetTensorMutableData<float>();
        std::vector<int64_t> sh = tensorShape(enc_outs[static_cast<size_t>(ei)]);
        erb_feat[name] = {p, std::move(sh)};
    }

    std::vector<Ort::Value> erb_inputs = buildInputs(erb_, memory_info_, erb_feat);
    if (erb_inputs.size() != erb_.input_names.size()) {
        DFN_PULSE_LOG("erb buildInputs: got %zu tensors, need %zu",
                      erb_inputs.size(), erb_.input_names.size());
        return false;
    }

    std::vector<const char *> erb_in_ptrs;
    std::vector<const char *> erb_out_ptrs;
    for (const auto &s : erb_.input_names)
        erb_in_ptrs.push_back(s.c_str());
    for (const auto &s : erb_.output_names)
        erb_out_ptrs.push_back(s.c_str());

    std::vector<Ort::Value> erb_outs;
    try {
        erb_outs = erb_.session->Run(Ort::RunOptions{nullptr},
                                     erb_in_ptrs.data(), erb_inputs.data(), erb_inputs.size(),
                                     erb_out_ptrs.data(), erb_out_ptrs.size());
    } catch (const Ort::Exception &e) {
        DFN_PULSE_LOG("erb Run: %s", e.what());
        return false;
    }

    updateStateFromOutputs(erb_, erb_outs);

    const int m_idx = indexOfName(erb_.output_names, "m");
    if (m_idx < 0) {
        DFN_PULSE_LOG("erb output 'm' not found");
        return false;
    }
    const size_t m_elems = shapeNumel(tensorShape(erb_outs[static_cast<size_t>(m_idx)]));
    if (m_elems > out_m_elems) {
        DFN_PULSE_LOG("mask m: model outputs %zu floats, buffer only %zu (enlarge m_buf)",
                      m_elems, out_m_elems);
        return false;
    }
    std::memcpy(out_m, erb_outs[static_cast<size_t>(m_idx)].GetTensorMutableData<float>(),
                m_elems * sizeof(float));

    std::unordered_map<std::string, std::pair<const float *, std::vector<int64_t>>> df_feat;
    for (const std::string &name : df_.feature_names) {
        const int ei = indexOfName(enc_.output_names, name);
        if (ei < 0) {
            DFN_PULSE_LOG("enc output for df missing '%s'", name.c_str());
            return false;
        }
        float *p = enc_outs[static_cast<size_t>(ei)].GetTensorMutableData<float>();
        std::vector<int64_t> sh = tensorShape(enc_outs[static_cast<size_t>(ei)]);
        df_feat[name] = {p, std::move(sh)};
    }

    std::vector<Ort::Value> df_inputs = buildInputs(df_, memory_info_, df_feat);
    if (df_inputs.size() != df_.input_names.size()) {
        DFN_PULSE_LOG("df buildInputs: got %zu tensors, need %zu",
                      df_inputs.size(), df_.input_names.size());
        return false;
    }

    std::vector<const char *> df_in_ptrs;
    std::vector<const char *> df_out_ptrs;
    for (const auto &s : df_.input_names)
        df_in_ptrs.push_back(s.c_str());
    for (const auto &s : df_.output_names)
        df_out_ptrs.push_back(s.c_str());

    std::vector<Ort::Value> df_outs;
    try {
        df_outs = df_.session->Run(Ort::RunOptions{nullptr},
                                   df_in_ptrs.data(), df_inputs.data(), df_inputs.size(),
                                   df_out_ptrs.data(), df_out_ptrs.size());
    } catch (const Ort::Exception &e) {
        DFN_PULSE_LOG("df Run: %s", e.what());
        return false;
    }

    updateStateFromOutputs(df_, df_outs);

    const int coef_idx = indexOfName(df_.output_names, "coefs");
    if (coef_idx < 0) {
        DFN_PULSE_LOG("df output 'coefs' not found");
        return false;
    }
    const size_t c_elems = shapeNumel(tensorShape(df_outs[static_cast<size_t>(coef_idx)]));
    if (c_elems > out_coefs_elems) {
        DFN_PULSE_LOG("coefs: model outputs %zu floats, buffer only %zu (enlarge coefs_buf)",
                      c_elems, out_coefs_elems);
        return false;
    }

    std::memcpy(out_coefs, df_outs[static_cast<size_t>(coef_idx)].GetTensorMutableData<float>(),
                c_elems * sizeof(float));

    return true;
}
