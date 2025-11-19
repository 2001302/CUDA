cmake_minimum_required(VERSION 3.15)
project(CUDA_Practice VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type (Debug/Release)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# MSVC compiler options (global)
if(MSVC)
    add_compile_options(/W4 /permissive-)
endif()

# Common library (shared utilities)
add_library(common STATIC
    src/common/logger.cpp
)

target_include_directories(common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# MSVC target-specific compile options
if(MSVC)
    target_compile_options(common PRIVATE
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2>
    )
    target_link_options(common PRIVATE
        $<$<CONFIG:Debug>:/DEBUG>
    )
endif()

# Example1 project
add_executable(example1
    src/example1/main.cpp
)

target_link_libraries(example1 PUBLIC common)

target_include_directories(example1 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# MSVC target-specific compile options
if(MSVC)
    target_compile_options(example1 PRIVATE
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2>
    )
    target_link_options(example1 PRIVATE
        $<$<CONFIG:Debug>:/DEBUG>
    )
endif()

