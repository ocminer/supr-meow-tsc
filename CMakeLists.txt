cmake_minimum_required(VERSION 3.18)
project(supr-meow-tsc LANGUAGES C CXX)
# CUDA joins the party for the GPU-native sampler pipeline (src/pow_gpu.cu):
# the PoI sampler's full-vocabulary sort + rank-bucket stats run as a fused
# CUB kernel instead of a ~10 ms/token host sort — the difference between the
# GPU idling behind the CPU and the GPU being the bottleneck (as it should be).
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc> $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
                      $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()
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

# These extensions are part of the audited native profile. Fail configuration
# rather than silently build a slower/different engine from pristine llama.
file(READ "${MEOW_LLAMA_DIR}/src/llama-context.cpp" MEOW_LLAMA_CONTEXT_SOURCE)
file(READ "${MEOW_LLAMA_DIR}/src/llama-kv-cache.cpp" MEOW_LLAMA_KV_SOURCE)
file(READ "${MEOW_LLAMA_DIR}/src/llama-cparams.h" MEOW_LLAMA_CPARAMS_SOURCE)
if(NOT MEOW_LLAMA_CONTEXT_SOURCE MATCHES "MEOW_KV_PAD64" OR
   NOT MEOW_LLAMA_CONTEXT_SOURCE MATCHES "MEOW_F16_KV_PAD64" OR
   NOT MEOW_LLAMA_KV_SOURCE MATCHES "MEOW_KV_SINGLE_UBATCH" OR
   NOT MEOW_LLAMA_CPARAMS_SOURCE MATCHES "#ifndef LLAMA_MAX_SEQ")
  message(FATAL_ERROR "Prepare the pinned llama dependency first: bash tools/prepare-llama.sh ${MEOW_LLAMA_DIR}")
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
set(MEOW_MAX_SEQUENCES 512 CACHE STRING "Compile-time llama sequence bound (experimental 1024)")
set_property(CACHE MEOW_MAX_SEQUENCES PROPERTY STRINGS 512 1024)
if(NOT MEOW_MAX_SEQUENCES MATCHES "^(512|1024)$")
  message(FATAL_ERROR "MEOW_MAX_SEQUENCES must be 512 or 1024")
endif()
target_compile_definitions(llama PUBLIC LLAMA_MAX_SEQ=${MEOW_MAX_SEQUENCES})

# ---------------------------------------------------------------------------
# Proof-of-inference sources.
#
# These live in the UMBRELLA repo (vendor/tensorcash/shared-utils/pow-utils),
# not in the llama.cpp fork — the fork's CMakeLists references pow_utils.cpp but
# does not contain it, because upstream's Dockerfile copies it in at image build
# time. Anyone cloning the fork alone therefore cannot build it. We reference
# the pinned llama submodule and the bundled TensorCash sources directly.
# FlatBuffers headers are regenerated with the installed flatc so they match
# the runtime headers used by the build.
# ---------------------------------------------------------------------------
set(MEOW_POW_DIR "${CMAKE_SOURCE_DIR}/vendor/tensorcash/shared-utils/pow-utils"
    CACHE PATH "TensorCash pow-utils checkout")
if(NOT EXISTS "${MEOW_POW_DIR}/pow_utils.cpp")
  message(FATAL_ERROR
    "proof-of-inference sources not found at ${MEOW_POW_DIR}.\n"
    "Run: git submodule update --init --recursive")
endif()

find_package(OpenSSL REQUIRED)
find_path(ARGON2_INCLUDE_DIR argon2.h)
find_library(ARGON2_LIBRARY NAMES argon2)
if(NOT ARGON2_INCLUDE_DIR OR NOT ARGON2_LIBRARY)
  message(FATAL_ERROR "libargon2 not found — install libargon2-dev. "
                      "It is REQUIRED: the chain is past v3 activation and every "
                      "proof needs the Argon2id admission puzzle.")
endif()
find_path(FLATBUFFERS_INCLUDE_DIR flatbuffers/flatbuffers.h)
if(NOT FLATBUFFERS_INCLUDE_DIR)
  message(FATAL_ERROR "flatbuffers headers not found — install FlatBuffers v25 "
                      "(the 2.x packaged by distros fails a hard version assert).")
endif()

# The *_generated.h checked into pow-utils were produced by flatc 23.5.26 and
# carry a hard static_assert against the runtime version, so they refuse to
# compile against any other FlatBuffers. Regenerate them from the .fbs schemas
# with the local flatc — which is exactly what upstream's Dockerfile does, and
# it keeps the build working across FlatBuffers releases.
find_library(ZMQ_LIBRARY NAMES zmq libzmq libzmq-mt-4_3_5 libzmq-mt-s-4_3_5)
find_path(ZMQ_INCLUDE_DIR zmq.h)
if(NOT ZMQ_LIBRARY OR NOT ZMQ_INCLUDE_DIR)
  message(FATAL_ERROR "libzmq not found — install libzmq3-dev. It arrives with "
                      "the MiningResponse serializer; the miner submits over "
                      "stratum and does not use the ZMQ egress.")
endif()

find_program(FLATC_EXECUTABLE flatc)
if(NOT FLATC_EXECUTABLE)
  message(FATAL_ERROR "flatc not found — needed to regenerate the proof schemas. "
                      "Install FlatBuffers v25 (compiler + headers).")
endif()
set(MEOW_FBS_DIR "${CMAKE_SOURCE_DIR}/vendor/tensorcash/shared-utils/fb-schemas"
    CACHE PATH "TensorCash flatbuffers schemas")
set(MEOW_FB_GEN "${CMAKE_BINARY_DIR}/fbgen")
file(MAKE_DIRECTORY ${MEOW_FB_GEN})
set(MEOW_FB_HEADERS
  ${MEOW_FB_GEN}/proof_generated.h
  ${MEOW_FB_GEN}/blockheader_generated.h
  ${MEOW_FB_GEN}/validation_generated.h)
add_custom_command(
  OUTPUT ${MEOW_FB_HEADERS}
  COMMAND ${FLATC_EXECUTABLE} --cpp -o ${MEOW_FB_GEN}
          ${MEOW_FBS_DIR}/proof.fbs ${MEOW_FBS_DIR}/blockheader.fbs ${MEOW_FBS_DIR}/validation.fbs
  DEPENDS ${MEOW_FBS_DIR}/proof.fbs ${MEOW_FBS_DIR}/blockheader.fbs ${MEOW_FBS_DIR}/validation.fbs
  COMMENT "Regenerating FlatBuffers proof schemas with ${FLATC_EXECUTABLE}")
add_custom_target(meow_fbgen DEPENDS ${MEOW_FB_HEADERS})

# Stage the sources into the build tree and drop the FRESH generated headers in
# beside them. Necessary because pow_utils.cpp uses a QUOTED include
# (#include "proof_generated.h"), which the compiler resolves next to the source
# file before any -I path — so include order alone cannot shadow the stale
# checked-in copies. Upstream's Dockerfile stages the same way.
set(MEOW_POI_STAGE ${CMAKE_BINARY_DIR}/poi)
file(MAKE_DIRECTORY ${MEOW_POI_STAGE})
set(MEOW_POI_STAGED
  ${MEOW_POI_STAGE}/pow_utils.cpp ${MEOW_POI_STAGE}/pow_v3.cpp
  ${MEOW_POI_STAGE}/bcred_table_r1024.cpp
  ${MEOW_POI_STAGE}/pow_zmq_writer.cpp
  # Proof v4: window-root derivation (E4) + the chain-bound Gumbel race draw.
  ${MEOW_POI_STAGE}/meow_pow_v4.cpp ${MEOW_POI_STAGE}/gumbel_sampler.cpp)
add_custom_command(
  OUTPUT ${MEOW_POI_STAGED}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MEOW_POW_DIR}/pow_utils.cpp ${MEOW_POW_DIR}/pow_utils.h
          ${MEOW_POW_DIR}/pow_v3.cpp ${MEOW_POW_DIR}/pow_v3.h
          ${MEOW_POW_DIR}/bcred_table_r1024.cpp ${MEOW_POW_DIR}/bcred_table_r1024.h
          ${MEOW_POW_DIR}/any_map_dump.h
          ${MEOW_POW_DIR}/pow_zmq_writer.cpp ${MEOW_POW_DIR}/pow_zmq_writer.h
          ${MEOW_POW_DIR}/pow_zmq_writer_helpers.h
          ${MEOW_POW_DIR}/meow_pow_v4.cpp ${MEOW_POW_DIR}/meow_pow_v4.h
          ${MEOW_POW_DIR}/gumbel_sampler.cpp ${MEOW_POW_DIR}/gumbel_sampler.h
          ${MEOW_POI_STAGE}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MEOW_FB_HEADERS} ${MEOW_POI_STAGE}
  # Fixes for upstream defects are applied to the STAGED copy, never to
  # vendor/, so the bundled source stays unchanged. See the script for what
  # each patch fixes; it fails the build if a target has disappeared.
  COMMAND ${CMAKE_COMMAND} -E env python3
          ${CMAKE_SOURCE_DIR}/tools/patch-staged-poi.py ${MEOW_POI_STAGE}
  COMMAND ${CMAKE_COMMAND} -E env python3
          ${CMAKE_SOURCE_DIR}/tools/optimize-staged-poi.py ${MEOW_POI_STAGE}
  DEPENDS ${CMAKE_SOURCE_DIR}/tools/optimize-staged-poi.py ${CMAKE_SOURCE_DIR}/tools/patch-staged-poi.py ${MEOW_FB_HEADERS} ${MEOW_POW_DIR}/pow_utils.cpp ${MEOW_POW_DIR}/pow_v3.cpp
  COMMENT "Staging proof-of-inference sources with regenerated schemas")

add_library(meow_poi STATIC ${MEOW_POI_STAGED})
# The Gumbel race's deterministic log (gumbel_sampler.cpp) is bit-exact ONLY
# with FP contraction OFF — it uses a shipped 129-entry log table and IEEE
# basic ops in a fixed order, NO libm, NO FMA. dlog_selftest() (called before
# every race) fails loudly if the TU was built with contraction/fast-math on.
set_source_files_properties(${MEOW_POI_STAGE}/gumbel_sampler.cpp PROPERTIES
  COMPILE_OPTIONS "$<IF:$<CXX_COMPILER_ID:MSVC>,/fp:strict,-ffp-contract=off>")
target_include_directories(meow_poi PUBLIC
  ${MEOW_POI_STAGE} ${CMAKE_SOURCE_DIR}/src ${ARGON2_INCLUDE_DIR} ${FLATBUFFERS_INCLUDE_DIR} ${ZMQ_INCLUDE_DIR})
target_link_libraries(meow_poi PUBLIC ${ARGON2_LIBRARY} ${ZMQ_LIBRARY} OpenSSL::Crypto)
# Linking libargon2 is NOT enough: the v3 admission code is behind this macro,
# and without it the sampler refuses to start on a v3 chain (its own guard —
# "a v3 miner without Argon2 would silently forfeit every admission-band
# window"). Define it wherever argon2 is linked.
target_compile_definitions(meow_poi PUBLIC POW_V3_HAVE_ARGON2)
# GPU sampler pipeline. POW_GPU_SORT=0 at runtime forces the CPU path; the
# CPU code remains intact as the fallback for both safety and non-CUDA builds.
target_sources(meow_poi PRIVATE ${CMAKE_SOURCE_DIR}/src/pow_gpu.cu)
target_sources(meow_poi PRIVATE ${CMAKE_SOURCE_DIR}/src/pow_histogram.cu)
target_compile_definitions(meow_poi PUBLIC POW_GPU_SORT_ENABLED)
set_source_files_properties(${CMAKE_SOURCE_DIR}/src/pow_gpu.cu PROPERTIES
  COMPILE_OPTIONS "--expt-relaxed-constexpr;-Xcompiler=$<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX2,-mavx2>")
set_target_properties(meow_poi PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# VDF (chiavdf). Sequential by design, so it stays on the CPU. Cannot be
# stubbed: a proof without a valid VDF is rejected by the network.
# ---------------------------------------------------------------------------
set(MEOW_CHIAVDF_DIR "${CMAKE_SOURCE_DIR}/vendor/tensorcash/shared-utils/chiavdf"
    CACHE PATH "chiavdf checkout")
if(NOT EXISTS "${MEOW_CHIAVDF_DIR}/src/verifier.h")
  message(FATAL_ERROR
    "chiavdf not found at ${MEOW_CHIAVDF_DIR}.\n"
    "Run: git submodule update --init --recursive")
endif()
find_path(GMP_INCLUDE_DIR gmp.h)
find_library(GMP_LIBRARY NAMES gmp)
find_library(GMPXX_LIBRARY NAMES gmpxx)
if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
  message(FATAL_ERROR "GMP not found — install libgmp-dev (chiavdf needs it).")
endif()

add_library(meow_vdf STATIC
  src/vdf.cpp
  ${MEOW_CHIAVDF_DIR}/src/refcode/lzcnt.c
)
target_include_directories(meow_vdf PUBLIC ${MEOW_CHIAVDF_DIR}/src ${GMP_INCLUDE_DIR})
target_link_libraries(meow_vdf PUBLIC ${GMP_LIBRARY})
if(GMPXX_LIBRARY)
  target_link_libraries(meow_vdf PUBLIC ${GMPXX_LIBRARY})
endif()
# chiavdf's fixed-point paths are only correct under truncation rounding and
# strict FP semantics; -ffast-math here would silently produce invalid proofs.
if(MSVC)
  target_compile_options(meow_vdf PRIVATE /fp:strict)
  target_sources(meow_vdf PRIVATE ${MEOW_CHIAVDF_DIR}/src/uint128_t/uint128_t.cpp)
else()
  target_compile_options(meow_vdf PRIVATE -fno-fast-math -frounding-math)
endif()
set_target_properties(meow_vdf PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(supr-meow-tsc
  src/main.cpp
  src/device.cpp
  src/stratum.cpp
  src/engine.cpp
  src/poi.cpp
  src/api.cpp
  src/canary.cpp
  src/tuning.cpp
  src/model_profile.cpp
)
target_include_directories(supr-meow-tsc PRIVATE
  ${MEOW_LLAMA_DIR}/include
  ${MEOW_LLAMA_DIR}/ggml/include
)
target_include_directories(supr-meow-tsc PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries(supr-meow-tsc PRIVATE meow_poi meow_vdf llama ggml CUDA::cudart CUDA::nvml Threads::Threads)

# A miner is copied between rigs, so prefer a binary that does not chase host
# libstdc++ versions. CUDA/NVML stay dynamic (they must match the driver).
option(MEOW_STATIC_LIBSTDCXX "Link libstdc++/libgcc statically" ON)
if(MEOW_STATIC_LIBSTDCXX AND NOT MSVC)
  target_link_options(supr-meow-tsc PRIVATE -static-libstdc++ -static-libgcc)
endif()

if(WIN32)
  target_link_libraries(supr-meow-tsc PRIVATE ws2_32 delayimp)
  target_link_options(supr-meow-tsc PRIVATE /DELAYLOAD:nvml.dll)
  if(TARGET ggml-cuda)
    target_link_libraries(ggml-cuda PRIVATE delayimp)
    target_link_options(ggml-cuda PRIVATE /DELAYLOAD:nvcuda.dll)
  endif()
endif()

option(MEOW_BUILD_TESTS "Build CPU proof and network contract tests" OFF)
if(MEOW_BUILD_TESTS)
  enable_testing()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_executable(meow-v4-root-test tests/v4_root_parity.cpp
    ${MEOW_POW_DIR}/meow_pow_v4.cpp ${MEOW_POW_DIR}/pow_v3.cpp ${MEOW_POW_DIR}/bcred_table_r1024.cpp)
  target_compile_definitions(meow-v4-root-test PRIVATE POW_V3_HAVE_ARGON2)
  target_include_directories(meow-v4-root-test PRIVATE ${MEOW_POW_DIR} vendor/nlohmann ${ARGON2_INCLUDE_DIR})
  target_link_libraries(meow-v4-root-test PRIVATE ${ARGON2_LIBRARY} OpenSSL::Crypto)
  add_test(NAME v4-root-parity COMMAND meow-v4-root-test ${CMAKE_SOURCE_DIR}/tests/vectors/v4_vectors.json)
  add_executable(meow-gumbel-test tests/gumbel_vectors_test.cpp ${MEOW_POW_DIR}/gumbel_sampler.cpp)
  set_source_files_properties(${MEOW_POW_DIR}/gumbel_sampler.cpp PROPERTIES
    COMPILE_OPTIONS "$<IF:$<CXX_COMPILER_ID:MSVC>,/fp:strict,-ffp-contract=off>")
  target_include_directories(meow-gumbel-test PRIVATE ${MEOW_POW_DIR} vendor/nlohmann)
  target_link_libraries(meow-gumbel-test PRIVATE OpenSSL::Crypto)
  add_test(NAME gumbel-parity COMMAND meow-gumbel-test ${CMAKE_SOURCE_DIR}/tests/vectors/gumbel_vectors.json)
  add_executable(meow-vdf-test tests/vdf_test.cpp)
  target_include_directories(meow-vdf-test PRIVATE src)
  target_link_libraries(meow-vdf-test PRIVATE meow_vdf)
  add_test(NAME vdf-proof-binding COMMAND meow-vdf-test ${CMAKE_SOURCE_DIR}/tests/vectors/vdf-reference.txt)
  add_executable(meow-network-test tests/network_client.cpp src/api.cpp src/stratum.cpp)
  target_include_directories(meow-network-test PRIVATE src)
  target_link_libraries(meow-network-test PRIVATE Threads::Threads)
  if(WIN32)
    target_link_libraries(meow-network-test PRIVATE ws2_32)
  endif()
  add_test(NAME network-contract COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/network_test.py $<TARGET_FILE:meow-network-test>)
endif()
install(TARGETS supr-meow-tsc RUNTIME DESTINATION bin)

# Offline measurement uses the production inference and proof sampler. It has
# no pool client: captured fixtures cannot accidentally become live work.
option(MEOW_BUILD_BENCH "Build the offline proof-generation benchmark" ON)
# The measurement/bench tool sources are dev-only and are excluded from the
# release Docker build context (.dockerignore), so guard the targets on their
# sources actually being present: a dev checkout builds them, the portable
# release build (which omits them) still configures cleanly instead of failing
# with "No SOURCES given to target".
if(MEOW_BUILD_BENCH AND EXISTS "${CMAKE_SOURCE_DIR}/tools/pow-bench.cpp")
  add_executable(meow-compact-check tools/compact-check.cpp)
  target_include_directories(meow-compact-check PRIVATE ${CMAKE_SOURCE_DIR}/src)
  add_executable(meow-histogram-check tools/histogram-check.cpp)
  target_link_libraries(meow-histogram-check PRIVATE meow_poi CUDA::cudart)
  add_executable(meow-replay-logits tools/replay-logits.cpp src/engine.cpp)
  target_include_directories(meow-replay-logits PRIVATE src vendor
    ${MEOW_LLAMA_DIR}/include ${MEOW_LLAMA_DIR}/ggml/include)
  target_link_libraries(meow-replay-logits PRIVATE meow_poi llama ggml CUDA::cudart Threads::Threads)
  add_executable(meow-pow-bench tools/pow-bench.cpp src/engine.cpp src/poi.cpp)
  target_include_directories(meow-pow-bench PRIVATE src vendor
    ${MEOW_LLAMA_DIR}/include ${MEOW_LLAMA_DIR}/ggml/include)
  target_link_libraries(meow-pow-bench PRIVATE meow_poi meow_vdf llama ggml
    CUDA::cudart Threads::Threads)
endif()
