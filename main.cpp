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

    size_t num_inputs = enc.GetInputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_name = enc.GetInputNameAllocated(i, allocator);  // ONNX API返回std::string（C++11+）
        std::cout << "输入 " << i << ": 名称='" << input_name.get() << "'" << std::endl;  // 替换LOGI，使用input_name直接（非.get()）
        auto type_info = enc.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
         std::cout << "输入 %zu: 名称='%s'" << i << input_name.get() << std::endl;

        auto shape = tensor_info.GetShape();
        std::string shape_str;
        for (auto dim : shape) {
            shape_str += std::to_string(dim) + " ";
        }
         std::cout << "  形状: [%s]" << shape_str.c_str() << std::endl;

        auto elem_type = tensor_info.GetElementType();
         std::cout << "  类型: %d (e.g., 1=float32)"<< elem_type << std::endl;
    }
}


int main() {
    CallbackPCMRecorder recorder;
    bool startResult = recorder.start();

    std::cout << " start aaudio recorder result is %d" << startResult << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(30));

    recorder.stop();

    return 0;
}