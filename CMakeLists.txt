cmake_minimum_required(VERSION 3.20)
project(NeuralForge VERSION 0.5.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(NEURALFORGE_BUILD_TESTS "Build unit tests" ON)
option(NEURALFORGE_BUILD_BENCHMARKS "Build benchmarks" ON)

set(NEURALFORGE_GTEST_GIT_TAG      "f8d7d77c06936315286eb55f8de22cd23c188571" CACHE STRING "GoogleTest git revision")
set(NEURALFORGE_BENCHMARK_GIT_TAG  "344117638c8ff7e239044fd0fa7085839fc03021" CACHE STRING "Google Benchmark git revision")

add_library(neuralforge
    src/tensor.cpp
    src/matmul.cpp
    src/nn_ops.cpp
    src/model.cpp
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

add_executable(neuralforge_model_runner apps/model_runner.cpp)
target_link_libraries(neuralforge_model_runner PRIVATE neuralforge)

if(NEURALFORGE_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        ${NEURALFORGE_GTEST_GIT_TAG}
        GIT_SHALLOW    TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(neuralforge_tests
        tests/test_tensor.cpp
        tests/test_inference.cpp
        tests/test_model.cpp
        tests/test_matmul_blocked.cpp
        tests/test_reliability.cpp
    )
    target_link_libraries(neuralforge_tests PRIVATE neuralforge GTest::gtest_main)
    target_compile_definitions(neuralforge_tests PRIVATE
        NEURALFORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )
    include(GoogleTest)
    gtest_discover_tests(neuralforge_tests)
endif()

if(NEURALFORGE_BUILD_BENCHMARKS)
    include(FetchContent)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        ${NEURALFORGE_BENCHMARK_GIT_TAG}
        GIT_SHALLOW    TRUE
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)

    add_executable(neuralforge_bench benchmarks/bench_matmul.cpp)
    target_link_libraries(neuralforge_bench PRIVATE neuralforge benchmark::benchmark)

    add_executable(neuralforge_bench_inference benchmarks/bench_inference.cpp)
    target_link_libraries(neuralforge_bench_inference PRIVATE neuralforge benchmark::benchmark)
    target_compile_definitions(neuralforge_bench_inference PRIVATE
        NEURALFORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )
endif()

message(STATUS "NeuralForge ${PROJECT_VERSION} configured")
message(STATUS "  C++ standard : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
