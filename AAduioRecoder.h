
#ifndef AAUDIORECORDER_AAUDIORECORDER_H
#define AAUDIORECORDER_AAUDIORECORDER_H
#include <fstream>
#include <iosfwd>
#include <thread>
#include <aaudio/AAudio.h>
#include <android/log.h>

#include "lwrb.h"
#include "kissfft/kiss_fftr.h"

// Buffer for 10 times of 20ms audio data
#define BUFFER_SIZE 960 * 10 * sizeof(float)

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NDKRecorder", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "NDKRecorder", __VA_ARGS__)


static const int ERB_BIN_RANGES[32][2] = {
    {0, 2}, {2, 4}, {4, 6}, {6, 8}, {8, 11}, {11, 14}, {14, 18}, {18, 22},
    {22, 27}, {27, 33}, {33, 40}, {40, 48}, {48, 57}, {57, 68}, {68, 81}, {81, 96},
    {96, 114}, {114, 135}, {135, 160}, {160, 190}, {190, 226}, {226, 269}, {269, 320},
    {320, 381}, {381, 453}, {453, 481},
    {481, 481}, {481, 481}, {481, 481}, {481, 481}, {481, 481}, {481, 481}
};

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
            return false;
        }

        result = AAudioStream_requestStart(stream);
        if (result != AAUDIO_OK) {
            LOGE("Failed to start stream");
            return false;
        }

        LOGI("Callback PCM recording started");

        pcmFile = fopen("/sdcard/test.pcm", "wb");
        if (!pcmFile) {
            LOGE("Failed to open PCM file");
            return false;
        }

        sourceFile = fopen("/sdcard/source.pcm", "wb");
        if (!sourceFile) {
            LOGE("Failed to open PCM file");
            return false;
        }

        fseek(pcmFile, 0, SEEK_END);
        long start_size = ftell(pcmFile);
        LOGI("Start file size: %ld bytes", start_size); // 应为 0

        running = true;
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
        LOGI("Callback PCM recording stopped");
    }

private:
    AAudioStream *stream;
    AAudioStreamBuilder *builder;
    static constexpr int SAMPLE_RATE = 48000;
    static constexpr int CHANNELS = 1;

    static constexpr int FRAME_MS = 20;
    static constexpr int FRAMES_PER_READ = SAMPLE_RATE * FRAME_MS / 1000; // 960
    static constexpr int BYTES_PER_READ = FRAMES_PER_READ * sizeof(float);

    static constexpr int FFT_SIZE = 960;
    static constexpr int HOP_SIZE = 480;
    static constexpr int FREQ_BINS = FFT_SIZE / 2 + 1;
    static constexpr int ERB_BANDS = 32;

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
        LOGI("dataCallback invoke audio nums %d", numFrames);
        auto *recorder = static_cast<CallbackPCMRecorder *>(userData);

        float *in = (float *) audioData;
        size_t free_space = lwrb_get_free(&recorder->audio_rb) / sizeof(float);
        size_t to_write = numFrames;

        if (to_write > free_space) {
            to_write = free_space;
            LOGE("Ring buffer overflow, dropping audio data free_space %zu", free_space);
        }

        LOGI("Ring buffer, to write audio data to_write %zu", to_write);

        if (to_write > 0) {
            lwrb_write(&recorder->audio_rb, (uint8_t *) in, to_write * sizeof(float));
            recorder->rb_cv.notify_one();
        }

        size_t free_space2 = lwrb_get_free(&recorder->audio_rb) / sizeof(float);

        LOGI("Ring buffer, free_space2 %zu", free_space2);

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
        LOGI("Handler got 20ms pcm, first sample=%f", pcm[0]);
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

            // iSTFT
            std::vector<float> time(FFT_SIZE);
            kiss_fftri(ifft_cfg, fft_out.data(), time.data());

            for (int i = 0; i < FFT_SIZE; ++i)
                ola_buffer[i] += (time[i] / FFT_SIZE) * window[i];

            // 输入 RMS（原始时域，对应这一 hop）
            float rms_in = 0.f;
            for (int i = 0; i < HOP_SIZE; ++i) {
                float v = input_buffer[i];   //
                rms_in += v * v;
            }
            rms_in = std::sqrt(rms_in / HOP_SIZE + 1e-8f);

            // 输出 RMS（增强后）
            float rms_out = 0.f;
            for (int i = 0; i < HOP_SIZE; ++i) {
                float v = ola_buffer[i];
                rms_out += v * v;
            }
            rms_out = std::sqrt(rms_out / HOP_SIZE + 1e-8f);

            // gain（官方逻辑）
            float gain = rms_in / rms_out;
            gain = std::clamp(gain, 0.1f, 10.0f);

            // 应用 gain
            for (int i = 0; i < HOP_SIZE; ++i) {
                ola_buffer[i] *= gain;
            }

            LOGI("RMS in=%.6f out=%.6f gain=%.3f", rms_in, rms_out, gain);


            float rms = 0.f;
            for (int i = 0; i < HOP_SIZE; ++i)
                rms += ola_buffer[i] * ola_buffer[i];

            rms = std::sqrt(rms / HOP_SIZE);
            LOGI("Output RMS = %.6f", rms);

            fwrite(ola_buffer.data(), sizeof(float), HOP_SIZE, pcmFile);

            memmove(ola_buffer.data(),
                    ola_buffer.data() + HOP_SIZE,
                    (FFT_SIZE - HOP_SIZE) * sizeof(float));
            memset(ola_buffer.data() + FFT_SIZE - HOP_SIZE, 0,
                   HOP_SIZE * sizeof(float));

            input_buffer.erase(input_buffer.begin(),
                               input_buffer.begin() + HOP_SIZE);
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

    void initSTFT() {
        window.resize(FFT_SIZE);

        float window_energy = 0.f;
        for (int i = 0; i < FFT_SIZE; ++i) {
            window[i] = 0.5f - 0.5f * cosf(2.f * M_PI * i / FFT_SIZE);
            window_energy += window[i] * window[i];
        }

        // COLA normalization (Hann + 50% overlap)
        ola_norm = window_energy / HOP_SIZE;

        fft_cfg  = kiss_fftr_alloc(FFT_SIZE, 0, nullptr, nullptr);
        ifft_cfg = kiss_fftr_alloc(FFT_SIZE, 1, nullptr, nullptr);

        input_buffer.reserve(FFT_SIZE * 2);
        ola_buffer.assign(FFT_SIZE, 0.f);

        fft_in.resize(FFT_SIZE);
        fft_out.resize(FREQ_BINS);
    }

    // =================== ONNX ===================
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "DFN"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> encSession;
    std::unique_ptr<Ort::Session> erbSession;
    std::unique_ptr<Ort::Session> dfSession;

    const char *enc_input_names[2] = {"feat_erb", "feat_spec"};
    const char *enc_output_names[7] = {"e0", "e1", "e2", "e3", "emb", "c0", "lsnr"};

    const char *erb_input_names[5] = {"emb", "e3", "e2", "e1", "e0"};
    const char *erb_output_names[1] = {"m"};

    const char *df_input_names[2] = {"emb", "c0"};
    const char *df_output_names[2] = {"coefs", "217"};
    // 状态
    Ort::Value e0{nullptr};
    Ort::Value e1{nullptr};
    Ort::Value e2{nullptr};
    Ort::Value e3{nullptr};

    static constexpr int NB_DF = 96;
    static constexpr int DF_ORDER = 5;

    // DF 历史：X[k][n-i]
    std::array<std::array<kiss_fft_cpx, DF_ORDER>, NB_DF> df_hist{};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    void initONNX() {
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        // 模型路径需修改为实际路径
        encSession = std::make_unique<Ort::Session>(env, "/data/local/tmp/enc.onnx", session_options);
        erbSession = std::make_unique<Ort::Session>(env, "/data/local/tmp/erb_dec.onnx", session_options);
        dfSession = std::make_unique<Ort::Session>(env, "/data/local/tmp/df_dec.onnx", session_options);
        LOGI("ONNX models loaded");
    }

    void computeERB(const std::vector<float> &log_pow, float *erb) {
        for (int b = 0; b < 32; ++b) {
            int s = ERB_BIN_RANGES[b][0];
            int e = ERB_BIN_RANGES[b][1];
            if (s >= e) {
                erb[b] = 0.f;
                continue;
            }
            float sum = 0.f;
            for (int k = s; k < e; ++k)
                sum += log_pow[k];
            erb[b] = sum / (e - s);
        }
    }

   void runDFN(std::vector<kiss_fft_cpx> &spectrum) {
    try {
        // ================= 1. 计算幅度和相位 + log1p(mag) =================
        std::vector<float> mag(FREQ_BINS);
        std::vector<float> phase(FREQ_BINS);

        for (int k = 0; k < FREQ_BINS; ++k) {
            float re = spectrum[k].r;
            float im = spectrum[k].i;
            mag[k] = std::hypot(re, im);
            phase[k] = std::atan2(im, re);
        }

        // ================= 2. ERB feature =================
        std::vector<float> log_pow(FREQ_BINS);
        for (int k = 0; k < FREQ_BINS; ++k) {
            log_pow[k] = std::log1p(mag[k]);  // 官方推荐
        }
        std::vector<float> feat_erb(ERB_BANDS, 0.0f);
        computeERB(log_pow, feat_erb.data());

        // ================= 3. Encoder 输入 =================
        constexpr int LOW_FREQ_BINS = 96;
        std::array<int64_t, 4> erb_shape{1, 1, 1, ERB_BANDS};
        std::array<int64_t, 4> spec_shape{1, 2, 1, LOW_FREQ_BINS};

        std::vector<float> spec_input(LOW_FREQ_BINS * 2);
        for (int k = 0; k < LOW_FREQ_BINS; ++k) {
            float p = mag[k] * mag[k];
            float lp = std::log1p(p);

            spec_input[k]                 = lp;
            spec_input[LOW_FREQ_BINS + k] = lp;
        }

        Ort::Value input_erb = Ort::Value::CreateTensor<float>(
            memory_info, feat_erb.data(), ERB_BANDS, erb_shape.data(), erb_shape.size());

        Ort::Value input_spec = Ort::Value::CreateTensor<float>(
            memory_info, spec_input.data(), LOW_FREQ_BINS * 2, spec_shape.data(), spec_shape.size());

        std::vector<Ort::Value> enc_inputs;
        enc_inputs.reserve(2);

        enc_inputs.emplace_back(std::move(input_erb));
        enc_inputs.emplace_back(std::move(input_spec));

        auto enc_outputs = encSession->Run(Ort::RunOptions{},
                                           enc_input_names, enc_inputs.data(), 2,
                                           enc_output_names, 7);

        Ort::Value new_e0   = std::move(enc_outputs[0]);
        Ort::Value new_e1   = std::move(enc_outputs[1]);
        Ort::Value new_e2   = std::move(enc_outputs[2]);
        Ort::Value new_e3   = std::move(enc_outputs[3]);
        Ort::Value new_emb  = std::move(enc_outputs[4]);
        Ort::Value new_c0   = std::move(enc_outputs[5]);
        // new_lsnr 可忽略

        // ================= 关键：深拷贝一份 emb 用于 DF decoder =================
        auto emb_info = new_emb.GetTensorTypeAndShapeInfo();
        auto emb_shape = emb_info.GetShape();
        size_t emb_elements = emb_info.GetElementCount();

        std::vector<float> emb_copy_data(emb_elements);
        std::memcpy(emb_copy_data.data(),
                    new_emb.GetTensorMutableData<float>(),
                    emb_elements * sizeof(float));

        Ort::Value emb_for_df = Ort::Value::CreateTensor<float>(
            memory_info, emb_copy_data.data(), emb_elements,
            emb_shape.data(), emb_shape.size());

        // ================= 4. ERB Decoder（先跑，优先使用原 emb） =================
        std::vector<Ort::Value> erb_inputs;
        erb_inputs.reserve(5);
        erb_inputs.push_back(std::move(new_emb));  // 原版 emb 给 ERB
        erb_inputs.push_back(std::move(new_e3));
        erb_inputs.push_back(std::move(new_e2));
        erb_inputs.push_back(std::move(new_e1));
        erb_inputs.push_back(std::move(new_e0));

        auto erb_outputs = erbSession->Run(Ort::RunOptions{},
                                           erb_input_names, erb_inputs.data(), 5,
                                           erb_output_names, 1);

        float* mask_ptr = erb_outputs[0].GetTensorMutableData<float>();


        // ================= 6. ERB gains 矩形上采样到 full_mask =================
        std::vector<float> full_mask(FREQ_BINS, 1.0f);  // 默认1.0

        static const int ERB_BIN_RANGES[32][2] = {
            {0, 2}, {2, 4}, {4, 6}, {6, 8}, {8, 11}, {11, 14}, {14, 18}, {18, 22},
            {22, 27}, {27, 33}, {33, 40}, {40, 48}, {48, 57}, {57, 68}, {68, 81}, {81, 96},
            {96, 114}, {114, 135}, {135, 160}, {160, 190}, {190, 226}, {226, 269}, {269, 320},
            {320, 381}, {381, 453}, {453, 481},
            {481, 481}, {481, 481}, {481, 481}, {481, 481}, {481, 481}, {481, 481}
        };

        for (int b = 0; b < ERB_BANDS; ++b) {
            int start = ERB_BIN_RANGES[b][0];
            int end   = ERB_BIN_RANGES[b][1];
            if (start >= end) continue;

            float g = std::clamp(mask_ptr[b], 0.0f, 2.0f);

            for (int k = start; k < end; ++k) {
                full_mask[k] = g;
            }
        }


        // ================= 5. DF Decoder（使用拷贝的 emb） =================
        std::vector<Ort::Value> df_inputs;
        df_inputs.reserve(2);
        df_inputs.push_back(std::move(emb_for_df));
        df_inputs.push_back(std::move(new_c0));

        auto df_outputs = dfSession->Run(Ort::RunOptions{},
                                         df_input_names, df_inputs.data(), 2,
                                         df_output_names, 1);

        float* df_coef = df_outputs[0].GetTensorMutableData<float>();

        // ================= 6. 应用 DF 低频增强 =================
        for (int k = 0; k < NB_DF; ++k) {
            kiss_fft_cpx y{0.0f, 0.0f};
            for (int i = 0; i <= DF_ORDER; ++i) {
                float a = df_coef[k * (DF_ORDER + 1) + i];
                const kiss_fft_cpx& x = (i == 0) ? spectrum[k] : df_hist[k][i - 1];
                y.r += a * x.r;
                y.i += a * x.i;
            }
            for (int i = DF_ORDER - 1; i > 0; --i) df_hist[k][i] = df_hist[k][i - 1];
            df_hist[k][0] = spectrum[k];
            spectrum[k] = y;
        }

        // ================= 7. 应用 ERB mask（全频段） =================
        for (int k = 0; k < FREQ_BINS; ++k) {
            spectrum[k].r *= full_mask[k];
            spectrum[k].i *= full_mask[k];
        }

        // ================= 8. Post-filter（推荐参数） =================
        // const float beta = 0.92f;
        // const float floor_gain = 0.1f;

        // for (int k = 0; k < FREQ_BINS; ++k) {
        //     float mag_new = std::hypot(spectrum[k].r, spectrum[k].i);
        //     float ph = std::atan2(spectrum[k].i, spectrum[k].r);
        //
        //     float gain = std::pow(mag_new, beta);
        //     gain = std::max(gain, floor_gain);
        //
        //     spectrum[k].r = gain * std::cos(ph);
        //     spectrum[k].i = gain * std::sin(ph);
        // }

        // ================= 9. 更新持久状态（仅 LSTM 隐状态） =================
        e0 = std::move(new_e0);
        e1 = std::move(new_e1);
        e2 = std::move(new_e2);
        e3 = std::move(new_e3);

    } catch (const Ort::Exception& e) {
        LOGE("ONNX Runtime error: %s", e.what());
        // 防止崩溃后状态混乱，可选：清空状态
        // e0 = Ort::Value{nullptr}; 等
    }
}
};


#endif //AAUDIORECORDER_AAUDIORECORDER_H
