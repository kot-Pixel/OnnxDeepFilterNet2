#ifndef AAUDIORECORDER_AAUDIORECORDER_H
#define AAUDIORECORDER_AAUDIORECORDER_H
#include <fstream>
#include <iosfwd>
#include <thread>
#include <aaudio/AAudio.h>
#include <android/log.h>

#include "lwrb.h"

// Buffer for 10 times of 20ms audio data
#define BUFFER_SIZE 960 * 10 * sizeof(float)

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NDKRecorder", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "NDKRecorder", __VA_ARGS__)


class CallbackPCMRecorder {
public:
    CallbackPCMRecorder() : stream(nullptr), builder(nullptr) {
       lwrb_init(&audio_rb, audio_rb_data, BUFFER_SIZE);
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
        float pcm[FRAMES_PER_READ];

        while (running) {
            std::unique_lock<std::mutex> lock(rb_mutex);

            rb_cv.wait(lock, [&] {
                return !running ||
                       lwrb_get_full(&audio_rb) >= BYTES_PER_READ;
            });

            if (!running) break;

            lwrb_read(&audio_rb,
                      (uint8_t*)pcm,
                      BYTES_PER_READ);

            lock.unlock();

            process20ms(pcm);
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


};



#endif //AAUDIORECORDER_AAUDIORECORDER_H