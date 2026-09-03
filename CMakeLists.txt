cmake_minimum_required(VERSION 3.20)
project(nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(NN_BUILD_EXAMPLES "Build example programs" ON)
option(NN_BUILD_BENCH "Build benchmarks" ON)

add_library(nn_flags INTERFACE)

# These are host-compiler flags. nvcc forwards options it does not recognise
# straight through to its host compiler (MSVC on Windows), so they must be
# scoped to CXX
set(NN_CXX "$<COMPILE_LANGUAGE:CXX>")
set(NN_CXX_RELEASE "$<AND:${NN_CXX},$<CONFIG:Release>>")
set(NN_CXX_DEBUG "$<AND:${NN_CXX},$<CONFIG:Debug>>")

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  target_compile_options(nn_flags INTERFACE
    $<${NN_CXX}:/W4;/permissive-;/EHsc>
    $<${NN_CXX_RELEASE}:/O2;/fp:precise>
    $<${NN_CXX_DEBUG}:/Od;/Zi;/fsanitize=address>)
else()
  target_compile_options(nn_flags INTERFACE
    $<${NN_CXX}:-Wall;-Wextra;-Wpedantic>
    $<${NN_CXX_RELEASE}:-O3;-march=native;-fno-math-errno>
    $<${NN_CXX_DEBUG}:-O0;-g>)

  if(WIN32)
    #target_compile_options(nn_flags INTERFACE $<${NN_CXX_DEBUG}:-fsanitize=address>)
    #target_link_options(nn_flags INTERFACE $<$<CONFIG:Debug>:-fsanitize=address>)
  else()
    target_compile_options(nn_flags INTERFACE
      $<${NN_CXX_DEBUG}:-fsanitize=address,undefined>)
    target_link_options(nn_flags INTERFACE
      $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
  endif()
endif()

# CUDA device code gets its own flags.
target_compile_options(nn_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)

# The CUDA backend is optional
find_program(NN_NVCC_EXECUTABLE nvcc
  DOC "nvcc, used only to choose the NN_WITH_CUDA default")
if(NN_NVCC_EXECUTABLE)
  set(NN_CUDA_DEFAULT ON)
else()
  set(NN_CUDA_DEFAULT OFF)
endif()
option(NN_WITH_CUDA "Build the CUDA backend" ${NN_CUDA_DEFAULT})
message(STATUS "CUDA backend: ${NN_WITH_CUDA}")

if(NN_WITH_CUDA)
  if(WIN32 AND NOT CMAKE_MT)
    get_filename_component(_nn_cxx_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    file(GLOB _nn_sdk_bin "C:/Program Files (x86)/Windows Kits/10/bin/*/x64")
    find_program(NN_MANIFEST_TOOL NAMES llvm-mt mt
      HINTS "${_nn_cxx_bin}" ${_nn_sdk_bin})
    if(NOT NN_MANIFEST_TOOL)
      message(FATAL_ERROR
        "No manifest tool found. Pass -DCMAKE_MT=<path to llvm-mt.exe or mt.exe>, "
        "or configure from a Visual Studio Developer Command Prompt.")
    endif()
    set(CMAKE_MT "${NN_MANIFEST_TOOL}" CACHE FILEPATH "Manifest tool" FORCE)
    message(STATUS "Manifest tool: ${CMAKE_MT}")
  endif()

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_RUNTIME_LIBRARY None)

  target_compile_definitions(nn_flags INTERFACE NN_WITH_CUDA)
endif()

file(GLOB_RECURSE NN_SOURCES CONFIGURE_DEPENDS src/*.cpp)
if(NN_WITH_CUDA)
  file(GLOB_RECURSE NN_CUDA_SOURCES CONFIGURE_DEPENDS src/*.cu)
  list(APPEND NN_SOURCES ${NN_CUDA_SOURCES})
endif()
add_library(nn STATIC ${NN_SOURCES})
target_include_directories(nn PUBLIC include)
target_include_directories(nn PRIVATE src third_party)
target_link_libraries(nn PUBLIC nn_flags)
if(NN_WITH_CUDA)
  target_link_libraries(nn PUBLIC CUDA::cudart CUDA::cublas)
  set_target_properties(nn PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

enable_testing()
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
if(NN_WITH_CUDA)
  file(GLOB TEST_CUDA_SOURCES CONFIGURE_DEPENDS tests/*.cu)
  list(APPEND TEST_SOURCES ${TEST_CUDA_SOURCES})
endif()
add_executable(test_all ${TEST_SOURCES})
target_link_libraries(test_all PRIVATE nn)
target_include_directories(test_all PRIVATE tests ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME unit COMMAND test_all)

if(NN_BUILD_EXAMPLES)
  file(GLOB EXAMPLE_DIRS CONFIGURE_DEPENDS
       RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/examples ${CMAKE_CURRENT_SOURCE_DIR}/examples/*)
  foreach(name ${EXAMPLE_DIRS})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/${name})
      file(GLOB srcs CONFIGURE_DEPENDS examples/${name}/*.cpp)
      add_executable(example_${name} ${srcs})
      target_link_libraries(example_${name} PRIVATE nn)
      target_compile_definitions(example_${name} PRIVATE
        NN_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
  endforeach()
endif()

if(NN_BUILD_BENCH)
  file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS bench/*.cpp)
  foreach(src ${BENCH_SOURCES})
    get_filename_component(name "${src}" NAME_WE)
    add_executable(${name} "${src}")
    target_link_libraries(${name} PRIVATE nn)
    target_include_directories(${name} PRIVATE bench ${CMAKE_CURRENT_SOURCE_DIR}/src)
  endforeach()
endif()