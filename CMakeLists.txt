cmake_minimum_required(VERSION 3.15)
project(CUDA_Practice VERSION 1.0.0 LANGUAGES CXX)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 빌드 타입 설정 (Debug/Release)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# 출력 디렉토리 설정
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# MSVC 컴파일러 옵션 (전역)
if(MSVC)
    add_compile_options(/W4 /permissive-)
endif()

# ============================================================================
# Common 라이브러리 (공용 유틸리티)
# ============================================================================
add_library(common STATIC
    src/common/Logger.cpp
)

target_include_directories(common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# MSVC 타겟별 컴파일 옵션
if(MSVC)
    target_compile_options(common PRIVATE
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2>
    )
    target_link_options(common PRIVATE
        $<$<CONFIG:Debug>:/DEBUG>
    )
endif()

# ============================================================================
# Example1 프로젝트
# ============================================================================
add_executable(example1
    src/example1/main.cpp
)

target_link_libraries(example1 PRIVATE common)

target_include_directories(example1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# MSVC 타겟별 컴파일 옵션
if(MSVC)
    target_compile_options(example1 PRIVATE
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2>
    )
    target_link_options(example1 PRIVATE
        $<$<CONFIG:Debug>:/DEBUG>
    )
endif()

