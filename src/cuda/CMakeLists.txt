# CUDA common library - shared CUDA utilities
# This library contains only .cu files
add_library(cuda STATIC
    simple.cu
)

# Explicitly set CUDA language for .cu files
set_source_files_properties(simple.cu PROPERTIES
    LANGUAGE CUDA
)

# Remove global compile options from CUDA target (MSVC options should go through -Xcompiler)
if(MSVC)
    # Remove global compile options that might interfere with nvcc
    get_target_property(CUDA_COMPILE_OPTIONS cuda COMPILE_OPTIONS)
    if(CUDA_COMPILE_OPTIONS)
        list(REMOVE_ITEM CUDA_COMPILE_OPTIONS "/W4" "/permissive-")
        set_target_properties(cuda PROPERTIES COMPILE_OPTIONS "${CUDA_COMPILE_OPTIONS}")
    endif()
    # Suppress C4819 warning (unicode characters in CUDA headers)
    # Use -Xcompiler to pass MSVC options to nvcc
    # Runtime library is set globally via CMAKE_MSVC_RUNTIME_LIBRARY
    target_compile_options(cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /wd4819>
    )
endif()

# Set CUDA architecture for CUDA library (CUDA 13.0 supports sm_75 and above)
# Visual Studio Generator: use single architecture to avoid nvcc command line issues
if(MSVC)
    # Use single architecture for Visual Studio to avoid nvcc command line parsing issues
    set_property(TARGET cuda PROPERTY CUDA_ARCHITECTURES "75")
else()
    set_property(TARGET cuda PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()

# Set include directories for header files
target_include_directories(cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# CUDA runtime is automatically linked when CUDA language is enabled

