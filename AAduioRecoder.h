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


class CallbackPCMRecorder {
public:
    CallbackPCMRecorder() : stream(nullptr), builder(nullptr) {
        lwrb_init(&audio_rb, audio_rb_data, BUFFER_SIZE);

        initSTFT(); 
    }

    bool read20ms(float* out_pcm) {
        size_t available = lwrb_get_full(&audio_rb);
        if (available < BYTES_PER_READ) {
            return false;
        }

        size_t read = lwrb_read(&audio_rb,
                                (uint8_t*)out_pcm,
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

        fseek(pcmFile, 0, SEEK_END);
        long start_size = ftell(pcmFile);
        LOGI("Start file size: %ld bytes", start_size);  // 应为 0

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
    AAudioStream* stream;
    AAudioStreamBuilder* builder;
    static constexpr int SAMPLE_RATE = 48000;
    static constexpr int CHANNELS = 1;

    static constexpr int FRAME_MS = 20;
    static constexpr int FRAMES_PER_READ = SAMPLE_RATE * FRAME_MS / 1000; // 960
    static constexpr int BYTES_PER_READ = FRAMES_PER_READ * sizeof(float);

    static constexpr int FFT_SIZE = 960;
    static constexpr int HOP_SIZE = 480;
    static constexpr int FREQ_BINS = FFT_SIZE / 2 + 1;

    std::thread handlerThread;
    std::atomic<bool> running{false};

    std::mutex rb_mutex;
    std::condition_variable rb_cv;

    // Ring buffer
    lwrb_t audio_rb;
    uint8_t audio_rb_data[BUFFER_SIZE];

    FILE* pcmFile = nullptr;

    static aaudio_data_callback_result_t dataCallback(AAudioStream* stream,void* userData,void* audioData,int32_t numFrames) {
        LOGI("dataCallback invoke audio nums %d", numFrames);
        auto* recorder = static_cast<CallbackPCMRecorder*>(userData);

        float* in = (float*)audioData;
        size_t free_space = lwrb_get_free(&recorder->audio_rb) / sizeof(float);
        size_t to_write = numFrames;

        if (to_write > free_space) {
            to_write = free_space;
            LOGE("Ring buffer overflow, dropping audio data free_space %zu", free_space);
        }

        LOGI("Ring buffer, to write audio data to_write %zu", to_write);

        if (to_write > 0) {
            lwrb_write(&recorder->audio_rb, (uint8_t*)in, to_write * sizeof(float));
            recorder->rb_cv.notify_one();
        }

        size_t free_space2 = lwrb_get_free(&recorder->audio_rb) / sizeof(float);

        LOGI("Ring buffer, free_space2 %zu", free_space2);

        return AAUDIO_CALLBACK_RESULT_CONTINUE;
    }

    static void errorCallback(
        AAudioStream* stream,
        void* userData,
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

            lwrb_read(&audio_rb, (uint8_t*)pcm.data(), BYTES_PER_READ);

            lock.unlock();

            processPCM(pcm.data(), FRAMES_PER_READ);
        }
    }

    void process20ms(const float* pcm) {
        LOGI("Handler got 20ms pcm, first sample=%f", pcm[0]);
        if (!pcmFile) return;

        size_t written = fwrite(pcm, sizeof(float), FRAMES_PER_READ, pcmFile);
        if (written != FRAMES_PER_READ) {
            LOGE("Failed to write all samples to file");
        }
    }

    void processPCM(const float* pcm, int n) {
        input_buffer.insert(input_buffer.end(), pcm, pcm + n);

        while (input_buffer.size() >= FFT_SIZE) {
            for (int i = 0; i < FFT_SIZE; ++i)
                fft_in[i] = input_buffer[i] * window[i];

            kiss_fftr(fft_cfg, fft_in.data(), fft_out.data());


            // iSTFT
            std::vector<float> time(FFT_SIZE);
            kiss_fftri(ifft_cfg, fft_out.data(), time.data());

            for (int i = 0; i < FFT_SIZE; ++i)
                ola_buffer[i] += (time[i] / FFT_SIZE) * window[i] / ola_norm;

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

};



#endif //AAUDIORECORDER_AAUDIORECORDER_H