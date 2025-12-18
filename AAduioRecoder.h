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

        return true;
    }

    void stop() {
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

    // Ring buffer
    lwrb_t audio_rb;
    uint8_t audio_rb_data[BUFFER_SIZE];

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


    void processAudio() {

    }

};

#endif //AAUDIORECORDER_AAUDIORECORDER_H