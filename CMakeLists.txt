cmake_minimum_required(VERSION 3.18)
project(supr-meow-tsc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Inference engine. Built from the pinned llama.cpp submodule so the miner is a
# single self-contained binary — no external server, no Python. The fork is used
# rather than upstream because it carries the proof-of-inference sampler.
# ---------------------------------------------------------------------------
set(MEOW_LLAMA_DIR "${CMAKE_SOURCE_DIR}/vendor/llama.cpp" CACHE PATH "llama.cpp checkout")
if(NOT EXISTS "${MEOW_LLAMA_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "llama.cpp not found at ${MEOW_LLAMA_DIR}.\n"
    "Run: git submodule update --init --recursive")
endif()

option(MEOW_CUDA "Build with CUDA (required to mine)" ON)
if(MEOW_CUDA)
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
add_subdirectory(${MEOW_LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama EXCLUDE_FROM_ALL)

add_executable(supr-meow-tsc
  src/main.cpp
  src/device.cpp
  src/stratum.cpp
  src/engine.cpp
)
target_include_directories(supr-meow-tsc PRIVATE
  ${MEOW_LLAMA_DIR}/include
  ${MEOW_LLAMA_DIR}/ggml/include
)
target_include_directories(supr-meow-tsc PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries(supr-meow-tsc PRIVATE llama ggml CUDA::cudart CUDA::nvml Threads::Threads)

# A miner is copied between rigs, so prefer a binary that does not chase host
# libstdc++ versions. CUDA/NVML stay dynamic (they must match the driver).
option(MEOW_STATIC_LIBSTDCXX "Link libstdc++/libgcc statically" ON)
if(MEOW_STATIC_LIBSTDCXX)
  target_link_options(supr-meow-tsc PRIVATE -static-libstdc++ -static-libgcc)
endif()

install(TARGETS supr-meow-tsc RUNTIME DESTINATION bin)
