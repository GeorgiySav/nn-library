cmake_minimum_required(VERSION 3.20)
project(nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_library(nn_flags INTERFACE)

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  target_compile_options(nn_flags INTERFACE
    /W4 /permissive- /EHsc
    $<$<CONFIG:Release>:/O2;/fp:precise>
    $<$<CONFIG:Debug>:/Od;/Zi;/fsanitize=address>)
else()
  target_compile_options(nn_flags INTERFACE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3;-march=native;-fno-math-errno>
    $<$<CONFIG:Debug>:-O0;-g>)

  if(WIN32)
    #target_compile_options(nn_flags INTERFACE $<$<CONFIG:Debug>:-fsanitize=address>)
    #target_link_options(nn_flags INTERFACE $<$<CONFIG:Debug>:-fsanitize=address>)
  else()
    target_compile_options(nn_flags INTERFACE
      $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    target_link_options(nn_flags INTERFACE
      $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
  endif()
endif()

file(GLOB_RECURSE NN_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(nn STATIC ${NN_SOURCES})
target_include_directories(nn PUBLIC include)
target_link_libraries(nn PUBLIC nn_flags)

enable_testing()
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
add_executable(test_all ${TEST_SOURCES})
target_link_libraries(test_all PRIVATE nn)
target_include_directories(test_all PRIVATE tests)
add_test(NAME unit COMMAND test_all)