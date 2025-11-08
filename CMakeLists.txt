cmake_minimum_required(VERSION 3.18)

# CUDA 경로 찾기 (project() 호출 전에 설정)
if(WIN32)
    find_program(CUDA_NVCC_EXECUTABLE
        NAMES nvcc.exe nvcc
        PATHS
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin"
        NO_DEFAULT_PATH
    )
    
    if(CUDA_NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER "${CUDA_NVCC_EXECUTABLE}")
        get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_NVCC_EXECUTABLE}" DIRECTORY)
        get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
        message(STATUS "Found CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
        
        # CUDA 컴파일러 정보를 명시적으로 설정하여 식별 단계 건너뛰기
        set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
        set(CMAKE_CUDA_COMPILER_VERSION "13.0")
        set(CMAKE_CUDA_COMPILER_FORCED TRUE)
    else()
        message(FATAL_ERROR "CUDA compiler (nvcc) not found. Please install CUDA Toolkit.")
    endif()
endif()

# CUDA 언어를 나중에 활성화 (식별 단계 건너뛰기)
project(CUDA_Practice LANGUAGES CXX)

# MSVC 링커 설정 (Ninja generator 사용 시)
if(MSVC AND CMAKE_GENERATOR STREQUAL "Ninja")
    # MSVC 런타임 라이브러리 링크
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # 링커 플래그 설정
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")
endif()

# CUDA 언어 활성화 (이미 컴파일러 정보를 설정했으므로 식별 단계 건너뛰기)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA 아키텍처 설정 (필요에 따라 변경)
# sm_75: Turing (RTX 20xx), sm_80: Ampere (RTX 30xx), sm_89: Ada (RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES "75;80;89" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Windows에서 MSVC 사용 시 CUDA 컴파일러 플래그 설정 (선택사항)
if(WIN32 AND MSVC)
    # UTF-8 인코딩 명시 (한글 주석/파일명 사용 시 유용)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /utf-8")
    # 코드 페이지 경고 억제
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /wd4819")
endif()

# 공통 include 디렉토리
include_directories(common)

# 빌드 출력 디렉토리 설정
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 서브디렉토리 추가 (새 프로젝트 추가 시 여기에 추가)
add_subdirectory(vector_add)

