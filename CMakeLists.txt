cmake_minimum_required(VERSION 3.18)
project(supr-meow-tsc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(supr-meow-tsc
  src/main.cpp
  src/device.cpp
)
target_link_libraries(supr-meow-tsc PRIVATE CUDA::cudart CUDA::nvml)

# A miner is copied between rigs, so prefer a binary that does not chase host
# libstdc++ versions. CUDA/NVML stay dynamic (they must match the driver).
option(MEOW_STATIC_LIBSTDCXX "Link libstdc++/libgcc statically" ON)
if(MEOW_STATIC_LIBSTDCXX)
  target_link_options(supr-meow-tsc PRIVATE -static-libstdc++ -static-libgcc)
endif()

install(TARGETS supr-meow-tsc RUNTIME DESTINATION bin)
