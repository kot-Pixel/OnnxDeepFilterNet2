#include <iostream>

#include <onnxruntime_cxx_api.h>


int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DeepFilterNet");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

    Ort::Session enc(env, "/data/local/tmp/enc.onnx", session_options);
    Ort::Session erb(env, "/data/local/tmp/erb_dec.onnx", session_options);
    Ort::Session df(env, "/data/local/tmp/df_dec.onnx", session_options);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    int frame_size = 480;      // 10ms @48kHz
    int hop_size   = 240;      // 50% overlap
    // std::vector<float> window = hann_window(frame_size);
    //
    // // 获取输入输出信息
    // size_t num_input_nodes = session.GetInputCount();
    // size_t num_output_nodes = session.GetOutputCount();
    //
    // std::cout << "Inputs: " << num_input_nodes << " Outputs: " << num_output_nodes << std::endl;
    //
    // // 假设是单输入 float[1, 3] 输出 float[1, 2]
    // std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
    // std::vector<int64_t> input_shape = {1, 3};
    //
    // Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    //     memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()
    // );
    //
    // auto output_tensors = session.Run(
    //     Ort::RunOptions{nullptr},
    //     &session.GetInputName(0, Ort::AllocatorWithDefaultOptions()),
    //     &input_tensor, 1,
    //     &session.GetOutputName(0, Ort::AllocatorWithDefaultOptions()), 1
    // );
    //
    // float* output_data = output_tensors[0].GetTensorMutableData<float>();
    std::cout << "Output onnx runtime ok...." << std::endl;
    return 0;
}