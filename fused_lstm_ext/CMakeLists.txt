cmake_minimum_required(VERSION 3.27)
project(_ext LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)
add_library(fused_lstm_lib)
target_sources(fused_lstm_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/fused_lstm/fused_lstm.cpp)
target_include_directories(fused_lstm_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(fused_lstm_lib PUBLIC mlx)
if(MLX_BUILD_METAL)
  mlx_build_metallib(TARGET fused_lstm_metallib TITLE fused_lstm_ext
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/fused_lstm/fused_lstm.metal
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_dependencies(fused_lstm_lib fused_lstm_metallib)
endif()
nanobind_add_module(_ext NB_STATIC STABLE_ABI LTO NOMINSIZE NB_DOMAIN mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE fused_lstm_lib)
if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
