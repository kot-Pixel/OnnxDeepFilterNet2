#include <iostream>

#include <onnxruntime_cxx_api.h>

#include "AAduioRecoder.h"

void testOnnxRuntime() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DeepFilterNet");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

    Ort::Session enc(env, "/data/local/tmp/enc.onnx", session_options);
    Ort::Session erb(env, "/data/local/tmp/erb_dec.onnx", session_options);
    Ort::Session df(env, "/data/local/tmp/df_dec.onnx", session_options);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    size_t enc_num_inputs = enc.GetInputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < enc_num_inputs; ++i) {
        auto input_name = enc.GetInputNameAllocated(i, allocator);
        std::cout << "enc.onnx Input " << i << ":- " << input_name << std::endl;
    }



    size_t enc_num_outputs = enc.GetOutputCount();
    for (size_t i = 0; i < enc_num_outputs; ++i) {
        auto out_name = enc.GetOutputNameAllocated(i, allocator);
        std::cout << "enc.onnx Output " << i << ":- " << out_name << std::endl;
    }

    size_t erb_num_inputs = erb.GetInputCount();
    for (size_t i = 0; i < erb_num_inputs; ++i) {
        auto input_name = erb.GetInputNameAllocated(i, allocator);
        std::cout << "erb.onnx Input " << i << ":- " << input_name << std::endl;
    }

    size_t erb_num_outputs = erb.GetOutputCount();
    for (size_t i = 0; i < erb_num_outputs; ++i) {
        auto out_name = erb.GetOutputNameAllocated(i, allocator);
        std::cout << "erb.onnx Output " << i << ":- " << out_name << std::endl;
    }


    size_t df_num_inputs = df.GetInputCount();
    for (size_t i = 0; i < df_num_inputs; ++i) {
        auto input_name = df.GetInputNameAllocated(i, allocator);
        std::cout << "df.onnx Input " << i << ":- " << input_name << std::endl;
    }

    size_t df_num_outputs = df.GetOutputCount();
    for (size_t i = 0; i < df_num_outputs; ++i) {
        auto out_name = df.GetOutputNameAllocated(i, allocator);
        std::cout << "df.onnx Output " << i << ":- " << out_name << std::endl;
    }
}


int main() {
    CallbackPCMRecorder recorder;
    bool startResult = recorder.start();

    std::cout << " start aaudio recorder result is %d" << startResult << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(30));

    recorder.stop();

    // testOnnxRuntime();

    return 0;
}