cmake_minimum_required(VERSION 3.22.1)
project(OnnxDeepFilterNet2)

set(CMAKE_CXX_STANDARD 17)

set(ONNX_RUNTIME_ROOT ${CMAKE_CURRENT_LIST_DIR}/onnx)
set(ONNX_RUNTIME_INCLUDE_ROOT ${ONNX_RUNTIME_ROOT}/include)
set(ONNX_RUNTIME_LIB_ROOT ${ONNX_RUNTIME_ROOT}/lib)

# 需使用编译时带 NNAPI 的 libonnxruntime.so（一般用官方 onnxruntime-android，非 onnxruntime-android-qnn）
# 配置示例: cmake ... -DONNX_ENABLE_NNAPI=ON -DONNX_ENABLE_QNN=OFF
option(ONNX_ENABLE_NNAPI "Append NNAPI Execution Provider before loading models" OFF)

# 需使用带 QNN EP 的 libonnxruntime.so（如 onnxruntime-android-qnn 中的 .so）；设备上需能加载 QNN SDK 的 libQnn*.so
option(ONNX_ENABLE_QNN "Append QNN Execution Provider (HTP); mutually exclusive with ONNX_ENABLE_NNAPI" OFF)

if(ONNX_ENABLE_QNN AND ONNX_ENABLE_NNAPI)
    message(FATAL_ERROR "Enable only one of ONNX_ENABLE_QNN or ONNX_ENABLE_NNAPI")
endif()


add_library(onnx_runtime SHARED IMPORTED)

set_target_properties(onnx_runtime PROPERTIES
    IMPORTED_LOCATION
        ${ONNX_RUNTIME_LIB_ROOT}/libonnxruntime.so

    INTERFACE_INCLUDE_DIRECTORIES
        ${ONNX_RUNTIME_INCLUDE_ROOT}
)

add_executable(OnnxDeepFilterNet2

    lwrb.c
    lwrb_ex.c
    kiss_fft.c
    kiss_fftr.c

    DeepFilterNet3OnnxPulse.cpp
    main.cpp
)


target_include_directories(
    OnnxDeepFilterNet2 PRIVATE

    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/kissfft
)

target_link_libraries(
        OnnxDeepFilterNet2 PRIVATE
        onnx_runtime

        aaudio
        log
)

# 安装/部署时与 libonnxruntime.so 同目录，运行时才能解析 OrtGetApiBase 等符号（否则 dlopen 会找到系统里错误/缺失的库）
set_target_properties(
    OnnxDeepFilterNet2 PROPERTIES
    INSTALL_RPATH "\$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

if(ONNX_ENABLE_NNAPI)
    target_compile_definitions(OnnxDeepFilterNet2 PRIVATE ONNX_ENABLE_NNAPI=1)
endif()

if(ONNX_ENABLE_QNN)
    target_compile_definitions(OnnxDeepFilterNet2 PRIVATE ONNX_ENABLE_QNN=1)
endif()

# 打开后 ORT 会输出 partition/EP 等到 logcat（tag 多为 onnxruntime）；调试完请关掉
option(ONNX_ORT_LOG_VERBOSE "Verbose ONNX Runtime logging for adb logcat" OFF)
if(ONNX_ORT_LOG_VERBOSE)
    target_compile_definitions(OnnxDeepFilterNet2 PRIVATE ONNX_ORT_LOG_VERBOSE=1)
endif()

install(TARGETS OnnxDeepFilterNet2
        RUNTIME DESTINATION bin
)
install(
    FILES ${ONNX_RUNTIME_LIB_ROOT}/libonnxruntime.so
    DESTINATION bin
)

# cmake -S. -Bbuild -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 -GNinja -DCMAKE_INSTALL_PREFIX=`pwd`/instal -DANDROID_STL=c++_static
# NNAPI: 追加 -DONNX_ENABLE_NNAPI=ON -DONNX_ENABLE_QNN=OFF，并确保 onnx/lib/libonnxruntime.so 为含 NNAPI 的构建
# 查看 NNAPI/partition 日志: 追加 -DONNX_ORT_LOG_VERBOSE=ON，adb logcat | grep -i onnxruntime
# cmake --install build --prefix instal  后把 instal/bin/ 下可执行文件与 libonnxruntime.so 一起 adb push；或: adb push .../libonnxruntime.so /data/local/tmp/ && export LD_LIBRARY_PATH=/data/local/tmp
# QNN EP: -DONNX_ENABLE_QNN=ON -DONNX_ENABLE_NNAPI=OFF  (use onnxruntime-android-qnn libonnxruntime.so). 设备上需从 QNN SDK 推送 libQnnSystem.so、libQnnHtp.so 等并设 LD_LIBRARY_PATH，或设 AAduioRecoder.h 里 kQnnHtpBackendSoPath
