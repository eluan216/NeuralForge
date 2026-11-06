cmake_minimum_required(VERSION 3.20)
project(NeuralForge VERSION 0.10.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(BUILD_SHARED_LIBS "Build a shared library (for the C ABI and ctypes)" OFF)
option(NEURALFORGE_ENABLE_SIMD "Enable guarded AVX2 / AArch64 NEON kernels" ON)
option(NEURALFORGE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(NEURALFORGE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
 add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
 add_link_options(-fsanitize=address,undefined)
endif()
option(NEURALFORGE_BUILD_TESTS "Build unit tests" ON)
option(NEURALFORGE_BUILD_BENCHMARKS "Build benchmarks" ON)

set(NEURALFORGE_GTEST_GIT_TAG      "f8d7d77c06936315286eb55f8de22cd23c188571" CACHE STRING "GoogleTest git revision")
set(NEURALFORGE_BENCHMARK_GIT_TAG  "344117638c8ff7e239044fd0fa7085839fc03021" CACHE STRING "Google Benchmark git revision")

add_library(neuralforge
    src/tensor.cpp
    src/matmul.cpp
    src/nn_ops.cpp
    src/model.cpp
    src/autograd.cpp
    src/loss.cpp
    src/optim.cpp
    src/kernels.cpp
    src/c_api.cpp
    src/training.cpp
    src/adaptation.cpp
    src/export_model.cpp
)

add_library(NeuralForge::neuralforge ALIAS neuralforge)
target_compile_features(neuralforge PUBLIC cxx_std_20)
if(NOT NEURALFORGE_ENABLE_SIMD)
 target_compile_definitions(neuralforge PRIVATE NEURALFORGE_DISABLE_SIMD=1)
endif()
if(BUILD_SHARED_LIBS)
 target_compile_definitions(neuralforge PUBLIC NF_SHARED PRIVATE NF_BUILDING)
 set_target_properties(neuralforge PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
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

add_executable(neuralforge_eval_accuracy apps/eval_accuracy.cpp)
target_link_libraries(neuralforge_eval_accuracy PRIVATE neuralforge)

add_executable(neuralforge_train_classifier apps/train_classifier.cpp)
target_link_libraries(neuralforge_train_classifier PRIVATE neuralforge)

add_executable(neuralforge_adapt_demo apps/adapt_demo.cpp)
target_link_libraries(neuralforge_adapt_demo PRIVATE neuralforge)
add_executable(neuralforge_latency apps/latency.cpp)
target_link_libraries(neuralforge_latency PRIVATE neuralforge)
add_executable(neuralforge_c_example examples/c_inference.c)
target_link_libraries(neuralforge_c_example PRIVATE neuralforge)
set_target_properties(neuralforge_c_example PROPERTIES LINKER_LANGUAGE CXX)

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
        tests/test_autograd.cpp
        tests/test_training.cpp
        tests/test_prepared.cpp
        tests/test_checkpoints.cpp
        tests/test_adaptation.cpp
        tests/test_graph_lifetime.cpp
        tests/test_classifier.cpp
    )
    find_package(Threads REQUIRED)
    target_link_libraries(neuralforge_tests PRIVATE neuralforge GTest::gtest_main Threads::Threads)
    target_compile_definitions(neuralforge_tests PRIVATE
        NEURALFORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )
    include(GoogleTest)
    gtest_discover_tests(neuralforge_tests)

    add_executable(neuralforge_allocation_test tests/test_allocations.cpp)
    target_link_libraries(neuralforge_allocation_test PRIVATE neuralforge)
    add_test(NAME PreparedNoAllocations COMMAND neuralforge_allocation_test "${CMAKE_SOURCE_DIR}/models/toy_classifier.nfmodel")
    add_test(NAME CEmbedding COMMAND neuralforge_c_example "${CMAKE_SOURCE_DIR}/models/toy_classifier.nfmodel")
    add_test(NAME ControlledAdaptation COMMAND neuralforge_adapt_demo --output "${CMAKE_BINARY_DIR}/adapt-artifacts")
    if(UNIX)
    add_test(
        NAME EvalDatasetValidation
        COMMAND bash "${CMAKE_SOURCE_DIR}/tests/test_eval_dataset.sh"
    )
    set_tests_properties(EvalDatasetValidation PROPERTIES
        ENVIRONMENT "BUILD_DIR=${CMAKE_BINARY_DIR};MODEL=${CMAKE_SOURCE_DIR}/models/toy_classifier.nfmodel"
        DEPENDS neuralforge_eval_accuracy
    )
    endif()
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

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS neuralforge EXPORT NeuralForgeTargets
 ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
 RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT NeuralForgeTargets FILE NeuralForgeTargets.cmake NAMESPACE NeuralForge::
 DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralForge)
configure_package_config_file(cmake/NeuralForgeConfig.cmake.in
 "${CMAKE_CURRENT_BINARY_DIR}/NeuralForgeConfig.cmake"
 INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralForge)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/NeuralForgeConfigVersion.cmake"
 VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/NeuralForgeConfig.cmake"
 "${CMAKE_CURRENT_BINARY_DIR}/NeuralForgeConfigVersion.cmake"
 DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralForge)
