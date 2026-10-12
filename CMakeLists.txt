cmake_minimum_required(VERSION 3.20)
project(NeuralForge VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(NEURALFORGE_BUILD_TESTS "Build unit tests" ON)
option(NEURALFORGE_BUILD_BENCHMARKS "Build benchmarks" ON)

add_library(neuralforge
    src/tensor.cpp
    src/matmul.cpp
    src/nn_ops.cpp
)

target_include_directories(neuralforge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(neuralforge PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_executable(neuralforge_demo apps/main.cpp)
target_link_libraries(neuralforge_demo PRIVATE neuralforge)

add_executable(neuralforge_inference_demo apps/inference_demo.cpp)
target_link_libraries(neuralforge_inference_demo PRIVATE neuralforge)

if(NEURALFORGE_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(neuralforge_tests
        tests/test_tensor.cpp
        tests/test_inference.cpp
    )
    target_link_libraries(neuralforge_tests PRIVATE neuralforge GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(neuralforge_tests)
endif()

if(NEURALFORGE_BUILD_BENCHMARKS)
    include(FetchContent)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)

    add_executable(neuralforge_bench benchmarks/bench_matmul.cpp)
    target_link_libraries(neuralforge_bench PRIVATE neuralforge benchmark::benchmark)
endif()

message(STATUS "NeuralForge ${PROJECT_VERSION} configured")
message(STATUS "  C++ standard : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
