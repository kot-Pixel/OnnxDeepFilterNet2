cmake_minimum_required(VERSION 3.22.1)
project(OnnxDeepFilterNet2)

set(CMAKE_CXX_STANDARD 17)

set(ONNX_RUNTIME_ROOT ${CMAKE_CURRENT_LIST_DIR}/onnx)
set(ONNX_RUNTIME_INCLUDE_ROOT ${ONNX_RUNTIME_ROOT}/include)
set(ONNX_RUNTIME_LIB_ROOT ${ONNX_RUNTIME_ROOT}/lib)

# 需使用编译时带 NNAPI 的 libonnxruntime.so；开启后会在 Session 上注册 NNAPI EP（NPU/GPU 等，依机型而定）
option(ONNX_ENABLE_NNAPI "Append NNAPI Execution Provider before loading models" OFF)

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

if(ONNX_ENABLE_NNAPI)
    target_compile_definitions(OnnxDeepFilterNet2 PRIVATE ONNX_ENABLE_NNAPI=1)
endif()

install(TARGETS OnnxDeepFilterNet2
        RUNTIME DESTINATION bin
)

# cmake -S. -Bbuild -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 -GNinja -DCMAKE_INSTALL_PREFIX=`pwd`/instal -DANDROID_STL=c++_static
