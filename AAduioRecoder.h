
#ifndef AAUDIORECORDER_AAUDIORECORDER_H
#define AAUDIORECORDER_AAUDIORECORDER_H
#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <fstream>
#include <iosfwd>
#include <thread>
#include <string>
#include <unordered_map>
#include <vector>
#include <aaudio/AAudio.h>
#include <android/log.h>

#include <memory>

#include "lwrb.h"
#include "kissfft/kiss_fftr.h"
#include "DeepFilterNet3OnnxPulse.h"
#include "df_features.h"

#if defined(__ANDROID__) && defined(ONNX_ENABLE_NNAPI)
#include "nnapi_provider_factory.h"
#endif
#if defined(__ANDROID__) && defined(ONNX_ENABLE_QNN)
#include "cpu_provider_factory.h"
#endif

// Buffer for 10 times of 20ms audio data
#define BUFFER_SIZE 960 * 10 * sizeof(float)

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "NDKRecorder", __VA_ARGS__)

class CallbackPCMRecorder {
public:
    CallbackPCMRecorder() : stream(nullptr), builder(nullptr) {
        lwrb_init(&audio_rb, audio_rb_data, BUFFER_SIZE);

        initSTFT();

        initONNX();
    }

    bool read20ms(float *out_pcm) {
        size_t available = lwrb_get_full(&audio_rb);
        if (available < BYTES_PER_READ) {
            return false;
        }

        size_t read = lwrb_read(&audio_rb,
                                (uint8_t *) out_pcm,
                                BYTES_PER_READ);
        return read == BYTES_PER_READ;
    }


    bool start() {
        aaudio_result_t result = AAudio_createStreamBuilder(&builder);
        if (result != AAUDIO_OK) {
            LOGE("Failed to create stream builder");
            return false;
        }

        AAudioStreamBuilder_setDirection(builder, AAUDIO_DIRECTION_INPUT);
        AAudioStreamBuilder_setSampleRate(builder, SAMPLE_RATE);
        AAudioStreamBuilder_setChannelCount(builder, CHANNELS);
        AAudioStreamBuilder_setFormat(builder, AAUDIO_FORMAT_PCM_FLOAT);
        AAudioStreamBuilder_setSharingMode(builder, AAUDIO_SHARING_MODE_SHARED);

        AAudioStreamBuilder_setDataCallback(builder, dataCallback, this);
        AAudioStreamBuilder_setErrorCallback(builder, errorCallback, this);

        result = AAudioStreamBuilder_openStream(builder, &stream);
        if (result != AAUDIO_OK) {
            LOGE("Failed to open stream");
            AAudioStreamBuilder_delete(builder);
            builder = nullptr;
            return false;
        }

        AAudioStreamBuilder_delete(builder);
        builder = nullptr;

        result = AAudioStream_requestStart(stream);
        if (result != AAUDIO_OK) {
            LOGE("Failed to start stream");
            AAudioStream_close(stream);
            stream = nullptr;
            return false;
        }

        pcmFile = fopen("/sdcard/test.pcm", "wb");
        if (!pcmFile) {
            LOGE("Failed to open PCM file");
            AAudioStream_requestStop(stream);
            AAudioStream_close(stream);
            stream = nullptr;
            return false;
        }

        sourceFile = fopen("/sdcard/source.pcm", "wb");
        if (!sourceFile) {
            LOGE("Failed to open PCM file");
            fclose(pcmFile);
            pcmFile = nullptr;
            AAudioStream_requestStop(stream);
            AAudioStream_close(stream);
            stream = nullptr;
            return false;
        }

        fseek(pcmFile, 0, SEEK_END);

        running = true;
        if (dfnPulse) {
            dfnPulse->reset();
        }
        df_hist = {};
        df::initMeanNormState(mean_norm_state_, ERB_BANDS);
        df::initUnitNormState(unit_norm_state_, NB_DF);

        handlerThread = std::thread(&CallbackPCMRecorder::handlerLoop, this);
        return true;
    }

    void stop() {
        running = false;
        rb_cv.notify_all();

        if (handlerThread.joinable()) {
            handlerThread.join();
        }

        if (pcmFile) {
            fclose(pcmFile);
            pcmFile = nullptr;
        }

        if (stream) {
            AAudioStream_requestStop(stream);
            AAudioStream_close(stream);
            stream = nullptr;
        }
        if (builder) {
            AAudioStreamBuilder_delete(builder);
            builder = nullptr;
        }
    }

private:
    AAudioStream *stream;
    AAudioStreamBuilder *builder;
    static constexpr int SAMPLE_RATE = 48000;
    static constexpr int CHANNELS = 1;

    static constexpr int FRAME_MS = 20;
    static constexpr int FRAMES_PER_READ = SAMPLE_RATE * FRAME_MS / 1000; // 960
    static constexpr int BYTES_PER_READ = FRAMES_PER_READ * sizeof(float);

    // 与训练 config.ini [df] 一致：sr=48000 fft_size=960 hop_size=480 nb_erb=32 nb_df=96 df_order=5
    static constexpr int FFT_SIZE = 960;
    static constexpr int HOP_SIZE = 480;
    static constexpr int FREQ_BINS = FFT_SIZE / 2 + 1;
    static constexpr int ERB_BANDS = 32;
    static constexpr int NB_DF = 96;
    static constexpr int DF_ORDER = 5;
    static constexpr float NORM_TAU = 1.f; // config [df] norm_tau
    /** 前若干 ERB 带（偏语音）掩码下界，稳住基频/共振峰（可略调 0.18～0.28） */
    static constexpr float MIN_ERB_MASK_VOICE = 0.24f;
    /** 其余偏高 ERB 带下界，可低于语音带以压嘶声/空气声（可略调 0.05～0.14） */
    static constexpr float MIN_ERB_MASK_HI = 0.09f;
    /** 带索引 < 该值用 MIN_ERB_MASK_VOICE，否则用 MIN_ERB_MASK_HI（32 带时约前 62% 偏语音） */
    static constexpr int ERB_VOICE_BANDS = 20;
    /** 时域干声混合；略低可减噪声渗入，略高更自然（可略调 0.06～0.14） */
    static constexpr float OUTPUT_DRY_MIX = 0.09f;
    /** 关：不写 test.pcm，测 CPU 时可关（省 fwrite）；关时仍会跑完整 ONNX+STFT */
    static constexpr bool kWriteProcessedPcmToFile = true;

    std::thread handlerThread;
    std::atomic<bool> running{false};

    std::mutex rb_mutex;
    std::condition_variable rb_cv;

    // Ring buffer
    lwrb_t audio_rb;
    uint8_t audio_rb_data[BUFFER_SIZE];

    FILE *pcmFile = nullptr;
    FILE *sourceFile = nullptr;

    static aaudio_data_callback_result_t dataCallback(AAudioStream *stream, void *userData, void *audioData,
                                                      int32_t numFrames) {
        auto *recorder = static_cast<CallbackPCMRecorder *>(userData);

        float *in = (float *) audioData;
        size_t free_space = lwrb_get_free(&recorder->audio_rb) / sizeof(float);
        size_t to_write = numFrames;

        if (to_write > free_space) {
            to_write = free_space;
            LOGE("Ring buffer overflow, dropping audio data free_space %zu", free_space);
        }

        if (to_write > 0) {
            lwrb_write(&recorder->audio_rb, (uint8_t *) in, to_write * sizeof(float));
            recorder->rb_cv.notify_one();
        }

        return AAUDIO_CALLBACK_RESULT_CONTINUE;
    }

    static void errorCallback(
        AAudioStream *stream,
        void *userData,
        aaudio_result_t error
    ) {
        LOGE("AAudio error: %d", error);
    }

    void handlerLoop() {
        std::vector<float> pcm(FRAMES_PER_READ);

        while (running) {
            std::unique_lock<std::mutex> lock(rb_mutex);

            rb_cv.wait(lock, [&] {
                return !running ||
                       lwrb_get_full(&audio_rb) >= BYTES_PER_READ;
            });

            if (!running) break;

            lwrb_read(&audio_rb, (uint8_t *) pcm.data(), BYTES_PER_READ);

            lock.unlock();

            process20ms(pcm.data());
            processPCM(pcm.data(), FRAMES_PER_READ);
        }
    }

    void process20ms(const float *pcm) {
        if (!sourceFile) return;

        size_t written = fwrite(pcm, sizeof(float), FRAMES_PER_READ, sourceFile);
        if (written != FRAMES_PER_READ) {
            LOGE("Failed to write all samples to file");
        }
    }

    void processPCM(const float *pcm, int n) {
        input_buffer.insert(input_buffer.end(), pcm, pcm + n);

        while (input_buffer.size() >= FFT_SIZE) {
            for (int i = 0; i < FFT_SIZE; ++i)
                fft_in[i] = input_buffer[i] * window[i];

            kiss_fftr(fft_cfg, fft_in.data(), fft_out.data());


            runDFN(fft_out);

            // iSTFT（复用 ifft_time_buf_）
            kiss_fftri(ifft_cfg, fft_out.data(), ifft_time_buf_.data());

            for (int i = 0; i < FFT_SIZE; ++i)
                ola_buffer[i] += (ifft_time_buf_[static_cast<size_t>(i)] / FFT_SIZE) * window[i];

            // 不要按 hop 做 rms_in/rms_out 增益：ola_buffer 前 HOP 是重叠相加结果，
            // 与 input_buffer 同一段的瞬时能量不可比，会泵动/失真。libDF 用 atten_lim/post_filter，不用该 gain。

            if (OUTPUT_DRY_MIX > 0.f) {
                for (int i = 0; i < HOP_SIZE; ++i) {
                    const float dry = input_buffer[static_cast<size_t>(i)];
                    ola_buffer[static_cast<size_t>(i)] =
                        ola_buffer[static_cast<size_t>(i)] * (1.f - OUTPUT_DRY_MIX) +
                        dry * OUTPUT_DRY_MIX;
                }
            }

            if (kWriteProcessedPcmToFile && pcmFile) {
                fwrite(ola_buffer.data(), sizeof(float), HOP_SIZE, pcmFile);
            }

            memmove(ola_buffer.data(),
                    ola_buffer.data() + HOP_SIZE,
                    (FFT_SIZE - HOP_SIZE) * sizeof(float));
            memset(ola_buffer.data() + FFT_SIZE - HOP_SIZE, 0,
                   HOP_SIZE * sizeof(float));

            // 用 memmove+resize 替代 erase，避免反复移动尾部未用容量时额外开销（相对 ONNX 仍很小）
            {
                const size_t n = input_buffer.size();
                std::memmove(input_buffer.data(), input_buffer.data() + static_cast<size_t>(HOP_SIZE),
                             (n - static_cast<size_t>(HOP_SIZE)) * sizeof(float));
                input_buffer.resize(n - static_cast<size_t>(HOP_SIZE));
            }
        }
    }


    // =================== STFT ===================
    kiss_fftr_cfg fft_cfg = nullptr;
    kiss_fftr_cfg ifft_cfg = nullptr;

    std::vector<float> window;
    std::vector<float> input_buffer;
    std::vector<float> ola_buffer;

    std::vector<float> fft_in;
    std::vector<kiss_fft_cpx> fft_out;

    float ola_norm = 1.f;

    /** 与 libDF erb_fb 一致；掩码上采样区间 */
    std::vector<int> erb_widths_;
    std::vector<std::pair<int, int>> erb_bin_ranges_;
    std::vector<float> mean_norm_state_;
    std::vector<float> unit_norm_state_;
    float norm_alpha_ = 0.f;
    float wnorm_ = 1.f;

    void initSTFT() {
        df::fillVorbisWindow(FFT_SIZE, window);

        float window_energy = 0.f;
        for (int i = 0; i < FFT_SIZE; ++i)
            window_energy += window[static_cast<size_t>(i)] * window[static_cast<size_t>(i)];
        ola_norm = window_energy / HOP_SIZE;

        erb_widths_ = df::erbFb(SAMPLE_RATE, FFT_SIZE, ERB_BANDS, 2);
        erb_bin_ranges_ = df::erbBinRanges(erb_widths_);
        df::initMeanNormState(mean_norm_state_, ERB_BANDS);
        df::initUnitNormState(unit_norm_state_, NB_DF);
        norm_alpha_ = df::normAlphaFromTau(SAMPLE_RATE, HOP_SIZE, NORM_TAU);
        wnorm_ = df::analysisWnorm(FFT_SIZE, HOP_SIZE);

        fft_cfg  = kiss_fftr_alloc(FFT_SIZE, 0, nullptr, nullptr);
        ifft_cfg = kiss_fftr_alloc(FFT_SIZE, 1, nullptr, nullptr);

        input_buffer.reserve(FFT_SIZE * 2);
        ola_buffer.assign(FFT_SIZE, 0.f);

        fft_in.resize(FFT_SIZE);
        fft_out.resize(FREQ_BINS);
        ifft_time_buf_.resize(static_cast<size_t>(FFT_SIZE));
    }

    // =================== ONNX（tract pulse 等价：见 DeepFilterNet3OnnxPulse） ===================
#if defined(__ANDROID__) && defined(ONNX_ENABLE_NNAPI)
    /** NNAPI_FLAG_CPU_DISABLED：与 NNAPI_FLAG_CPU_ONLY 互斥。true = 禁止 NNAPI 走其自带 CPU 实现，改由 ORT CPU 算子承接分区 */
    static constexpr bool kNnapiCpuDisabled = false;
#endif
#if defined(__ANDROID__) && defined(ONNX_ENABLE_QNN)
    /**
     * QNN EP 依赖高通 QNN SDK 的 .so（与 ORT 分开部署），例如从 SDK 的 android/arm64-v8a 拷到设备：
     *   libQnnSystem.so、libQnnHtp.so，及文档要求的其它依赖（如 libQnnCpu.so 等，以当前 QNN 版本为准）。
     * 部署方式二选一：
     *   (1) 全部放在同一目录，并 export LD_LIBRARY_PATH=该目录；
     *   (2) 设 kQnnHtpBackendSoPath 为设备上 libQnnHtp.so 的绝对路径（与 backend_type 互斥，见 ORT 文档）。
     */
    static constexpr const char *kQnnHtpArch = "68";
    /** 非空则使用 QNN `backend_path` 指向该文件；为空则用 `backend_type=htp`（依赖 LD_LIBRARY_PATH 能找到 libQnnHtp.so） */
    static constexpr const char *kQnnHtpBackendSoPath = "";
#endif
#if defined(ONNX_ORT_LOG_VERBOSE) && ONNX_ORT_LOG_VERBOSE
    Ort::Env env{ORT_LOGGING_LEVEL_VERBOSE, "DFN"};
#else
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "DFN"};
#endif
    Ort::SessionOptions session_options;
    std::unique_ptr<DeepFilterNet3OnnxPulse> dfnPulse;

    // DF 历史：X[k][n-i]
    std::array<std::array<kiss_fft_cpx, DF_ORDER>, NB_DF> df_hist{};

    /** runDFN / iSTFT 复用缓冲，避免每帧 heap 分配 */
    std::vector<float> feat_erb_buf_;
    std::vector<float> spec_input_buf_;
    std::vector<float> onnx_m_buf_;
    std::vector<float> onnx_coefs_buf_;
    std::vector<float> full_mask_buf_;
    std::vector<kiss_fft_cpx> df_out_buf_;
    std::vector<float> ifft_time_buf_;

    void allocateDfnBuffers() {
        if (!dfnPulse)
            return;
        size_t n_m = dfnPulse->erbMaskElementCount();
        size_t n_c = dfnPulse->dfCoefsElementCount();
        if (n_m == 0)
            n_m = static_cast<size_t>(ERB_BANDS);
        if (n_c == 0)
            n_c = static_cast<size_t>(NB_DF) * static_cast<size_t>(DF_ORDER) * 2u;
        n_m = std::max(n_m, static_cast<size_t>(ERB_BANDS) * 4u);
        n_c = std::max(n_c, static_cast<size_t>(NB_DF) * static_cast<size_t>(DF_ORDER) * 2u);
        onnx_m_buf_.resize(n_m);
        onnx_coefs_buf_.resize(n_c);
        feat_erb_buf_.resize(static_cast<size_t>(ERB_BANDS));
        spec_input_buf_.resize(static_cast<size_t>(NB_DF * 2));
        full_mask_buf_.resize(static_cast<size_t>(FREQ_BINS));
        df_out_buf_.resize(static_cast<size_t>(NB_DF));
    }

    void initONNX() {
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#if defined(ONNX_ORT_LOG_VERBOSE) && ONNX_ORT_LOG_VERBOSE
        session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
#endif

#if defined(__ANDROID__) && defined(ONNX_ENABLE_QNN)
        // 需 libonnxruntime 带 QNN EP；且设备上能 dlopen QNN SDK（缺 libQnnHtp.so 会 SetupBackend failed）
        try {
            std::unordered_map<std::string, std::string> qnn_options;
            if (kQnnHtpBackendSoPath[0] != '\0') {
                qnn_options["backend_path"] = kQnnHtpBackendSoPath;
            } else {
                qnn_options["backend_type"] = "htp";
            }
            qnn_options["htp_arch"] = kQnnHtpArch;
            session_options.AppendExecutionProvider("QNN", qnn_options);
            {
                Ort::Status cpu_status{OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 1)};
                if (!cpu_status.IsOK()) {
                    LOGE("CPU EP append failed: %s", cpu_status.GetErrorMessage().c_str());
                }
            }
        } catch (const Ort::Exception &e) {
            LOGE("QNN EP append failed (fallback implicit CPU): %s", e.what());
        }
#elif defined(__ANDROID__) && defined(ONNX_ENABLE_NNAPI)
        // 先注册 NNAPI；需 libonnxruntime 编译含 NNAPI。可与 NNAPI_FLAG_USE_FP16 等按位或组合。
        {
            uint32_t nnapi_flags = static_cast<uint32_t>(NNAPI_FLAG_USE_NONE);
            if (kNnapiCpuDisabled) {
                nnapi_flags |= static_cast<uint32_t>(NNAPI_FLAG_CPU_DISABLED);
            }
            Ort::Status nnapi_status{OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, nnapi_flags)};
            if (!nnapi_status.IsOK()) {
                LOGE("NNAPI EP append failed (fallback CPU): %s",
                     nnapi_status.GetErrorMessage().c_str());
            }
        }
#endif

        DeepFilterNet3OnnxPulse::Config cfg;
        cfg.enc_path = "/data/local/tmp/enc.onnx";
        cfg.erb_path = "/data/local/tmp/erb_dec.onnx";
        cfg.df_path = "/data/local/tmp/df_dec.onnx";
        cfg.nb_erb = ERB_BANDS;
        cfg.nb_df = NB_DF;
        cfg.df_order = DF_ORDER;
        try {
            dfnPulse = std::make_unique<DeepFilterNet3OnnxPulse>(env, session_options, cfg);
            allocateDfnBuffers();
        } catch (const Ort::Exception &e) {
            LOGE("ONNX load error (ORT): %s", e.what());
        } catch (const std::exception &e) {
            LOGE("ONNX load error: %s", e.what());
        }
    }

    void runDFN(std::vector<kiss_fft_cpx> &spectrum) {
        if (!dfnPulse) {
            return;
        }
        if (onnx_m_buf_.empty() || onnx_coefs_buf_.empty()) {
            allocateDfnBuffers();
            if (onnx_m_buf_.empty())
                return;
        }

        // libDF analysis：FFT 后乘 wnorm，与训练/ONNX 一致
        for (int k = 0; k < FREQ_BINS; ++k) {
            spectrum[static_cast<size_t>(k)].r *= wnorm_;
            spectrum[static_cast<size_t>(k)].i *= wnorm_;
        }

        df::featErb(erb_widths_, spectrum.data(), FREQ_BINS, norm_alpha_, mean_norm_state_,
                    feat_erb_buf_.data(), ERB_BANDS);

        df::featSpecUnitNorm(spectrum.data(), NB_DF, norm_alpha_, unit_norm_state_, spec_input_buf_.data());

        if (!dfnPulse->runFrame(feat_erb_buf_.data(), spec_input_buf_.data(),
                                onnx_m_buf_.data(), onnx_m_buf_.size(),
                                onnx_coefs_buf_.data(), onnx_coefs_buf_.size())) {
            LOGE("DeepFilterNet pulse runFrame failed (see logcat tag OnnxPulse)");
            const float inv = 1.f / wnorm_;
            for (int k = 0; k < FREQ_BINS; ++k) {
                spectrum[static_cast<size_t>(k)].r *= inv;
                spectrum[static_cast<size_t>(k)].i *= inv;
            }
            return;
        }

        float *mask_ptr = onnx_m_buf_.data();
        float *df_coef = onnx_coefs_buf_.data();

        std::fill(full_mask_buf_.begin(), full_mask_buf_.end(), 1.f);

        const int mask_bands =
            static_cast<int>(std::min<size_t>(onnx_m_buf_.size(), static_cast<size_t>(ERB_BANDS)));
        for (int b = 0; b < mask_bands; ++b) {
            const int start = erb_bin_ranges_[static_cast<size_t>(b)].first;
            const int end = erb_bin_ranges_[static_cast<size_t>(b)].second;
            if (start >= end)
                continue;
            const float floor_g =
                (b < ERB_VOICE_BANDS) ? MIN_ERB_MASK_VOICE : MIN_ERB_MASK_HI;
            const float g = std::max(std::clamp(mask_ptr[b], 0.0f, 2.0f), floor_g);
            for (int k = start; k < end; ++k)
                full_mask_buf_[static_cast<size_t>(k)] = g;
        }

        // 与 libDF tract.rs::df 一致：df_order 个复数抽头；时间顺序最旧帧 × coef[:,:,0]，当前帧 × coef[:,:,df_order-1]
        // ONNX 常见 runtime shape [1,1,nb_df,df_order*2]（如 [1,1,96,10]）：最后一维为 5 复数展平，bin k 偏移 k*10+tap*2
        for (int k = 0; k < NB_DF; ++k) {
            kiss_fft_cpx y{0.0f, 0.0f};
            for (int i = 0; i < DF_ORDER; ++i) {
                const kiss_fft_cpx s =
                    (i == DF_ORDER - 1) ? spectrum[static_cast<size_t>(k)]
                                        : df_hist[static_cast<size_t>(k)][static_cast<size_t>(DF_ORDER - 2 - i)];
                const size_t c0 = static_cast<size_t>(k) * static_cast<size_t>(DF_ORDER) * 2u
                                  + static_cast<size_t>(i) * 2u;
                const float cr = df_coef[c0];
                const float ci = df_coef[c0 + 1];
                y.r += s.r * cr - s.i * ci;
                y.i += s.r * ci + s.i * cr;
            }
            df_out_buf_[static_cast<size_t>(k)] = y;
            for (int j = DF_ORDER - 1; j > 0; --j)
                df_hist[static_cast<size_t>(k)][static_cast<size_t>(j)] =
                    df_hist[static_cast<size_t>(k)][static_cast<size_t>(j - 1)];
            df_hist[static_cast<size_t>(k)][0] = spectrum[static_cast<size_t>(k)];
        }

        // tract：先对延迟帧乘 ERB 掩码，再用 DF 输出覆盖前 nb_df 个 bin（不再对低频乘掩码）
        for (int k = 0; k < FREQ_BINS; ++k) {
            spectrum[static_cast<size_t>(k)].r *= full_mask_buf_[static_cast<size_t>(k)];
            spectrum[static_cast<size_t>(k)].i *= full_mask_buf_[static_cast<size_t>(k)];
        }
        for (int k = 0; k < NB_DF; ++k) {
            spectrum[static_cast<size_t>(k)] = df_out_buf_[static_cast<size_t>(k)];
        }

        const float inv_wnorm = 1.f / wnorm_;
        for (int k = 0; k < FREQ_BINS; ++k) {
            spectrum[static_cast<size_t>(k)].r *= inv_wnorm;
            spectrum[static_cast<size_t>(k)].i *= inv_wnorm;
        }
    }
};


#endif //AAUDIORECORDER_AAUDIORECORDER_H
