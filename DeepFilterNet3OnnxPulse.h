#ifndef DEEPFILTERNET3_ONNX_PULSE_H
#define DEEPFILTERNET3_ONNX_PULSE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * DeepFilterNet3 三子图（enc / erb_dec / df_dec）的 ONNX Runtime 推理封装。
 * 对除「每帧特征」以外的输入自动视为 pulse 状态：首帧清零，之后每帧将上一帧
 * 配对输出拷贝回对应输入（与 libDF tract 的 pulse 行为等价）。
 *
 * 特征输入名（与官方 export.py 一致）：
 *   enc:   feat_erb, feat_spec
 *   erb:   emb, e3, e2, e1, e0
 *   df:    emb, c0
 *
 * 若模型另有 state_* / out_state_* 等，通过名称启发式配对；若无额外状态张量则退化为
 * 与仅前馈子图相同的行为。
 *
 * 与训练 config.ini [df] 对齐的默认值：
 *   sr=48000, fft_size=960, hop_size=480, nb_erb=32, nb_df=96, df_order=5, df_lookahead=0
 * feat_spec 张量形状 [1, 2, 1, nb_df]（实部/虚部两路 log-power，与 export.py 一致）。
 */
class DeepFilterNet3OnnxPulse {
public:
    struct Config {
        const char *enc_path = nullptr;
        const char *erb_path = nullptr;
        const char *df_path = nullptr;
        /** [df] nb_erb — ERB 带数，feat_erb 最后一维 */
        int nb_erb = 32;
        /** [df] nb_df — 深度滤波频率 bin 数，feat_spec 最后一维 */
        int nb_df = 96;
        /** [df] df_order — 与 libDF 一致；ONNX 常展平为 [*,*,nb_df,df_order*2]（如 [1,1,96,10]） */
        int df_order = 5;
    };

    DeepFilterNet3OnnxPulse(Ort::Env &env, Ort::SessionOptions &session_options, const Config &cfg);

    void reset();

    /**
     * 处理一帧。feat_erb / feat_spec 布局与官方 df_features 一致（与现有 AAduioRecoder 相同）。
     * out_m / out_coefs 由调用方提供缓冲区；元素个数须与模型输出一致（否则返回 false）。
     */
    bool runFrame(const float *feat_erb, const float *feat_spec,
                  float *out_m, size_t out_m_elems,
                  float *out_coefs, size_t out_coefs_elems);

    /** 分配缓冲区用：max(元数据 numel, config 下限)；动态维元数据常低估，下限为 nb_erb / (nb_df*df_order*2) */
    size_t erbMaskElementCount() const { return erb_m_numel_; }
    size_t dfCoefsElementCount() const { return df_coefs_numel_; }

    int nbErb() const { return nb_erb_; }
    int nbDf() const { return nb_df_; }
    int dfOrder() const { return df_order_; }

private:
    struct Subgraph {
        std::unique_ptr<Ort::Session> session;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::unordered_set<std::string> feature_names;

        std::unordered_map<std::string, std::vector<float>> state_buffers;
        std::unordered_map<std::string, std::vector<int64_t>> state_shapes;
        /** 状态输入名 -> 上一帧应写入该输入的输出名（空表示无法配对，跳过写回） */
        std::unordered_map<std::string, std::string> state_in_to_out;
    };

    static Subgraph loadSubgraph(Ort::Env &env, Ort::SessionOptions &opt, const char *path,
                                 std::unordered_set<std::string> feature_names);

    static void resetSubgraph(Subgraph &g);
    static std::vector<int64_t> tensorShape(const Ort::Value &v);
    static size_t shapeNumel(const std::vector<int64_t> &shape);

    static std::string guessStateOutputForInput(const std::string &in,
                                                const std::vector<std::string> &outputs);

    static std::vector<Ort::Value> buildInputs(Subgraph &g, Ort::MemoryInfo &mem,
        const std::unordered_map<std::string, std::pair<const float *, std::vector<int64_t>>> &features);

    static void updateStateFromOutputs(Subgraph &g, std::vector<Ort::Value> &outs);

    static int indexOfName(const std::vector<std::string> &names, const std::string &n);

    Ort::MemoryInfo memory_info_;
    Subgraph enc_;
    Subgraph erb_;
    Subgraph df_;

    size_t erb_m_numel_ = 0;
    size_t df_coefs_numel_ = 0;

    int nb_erb_ = 32;
    int nb_df_ = 96;
    int df_order_ = 5;

    static size_t queryOutputFloatNumel(const Subgraph &g, const char *output_name);
};

#endif
