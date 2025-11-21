# CUDA library - compile CUDA files separately
add_library(example2_cuda STATIC
    simple.cu
)

# Explicitly set CUDA language for .cu files
set_source_files_properties(simple.cu PROPERTIES
    LANGUAGE CUDA
)

# Remove global compile options from CUDA target (MSVC options should go through -Xcompiler)
if(MSVC)
    # Remove global compile options that might interfere with nvcc
    get_target_property(CUDA_COMPILE_OPTIONS example2_cuda COMPILE_OPTIONS)
    if(CUDA_COMPILE_OPTIONS)
        list(REMOVE_ITEM CUDA_COMPILE_OPTIONS "/W4" "/permissive-")
        set_target_properties(example2_cuda PROPERTIES COMPILE_OPTIONS "${CUDA_COMPILE_OPTIONS}")
    endif()
endif()

# Set CUDA architecture for CUDA library (CUDA 13.0 supports sm_75 and above)
# Visual Studio Generator: use single architecture to avoid nvcc command line issues
if(MSVC)
    # Use single architecture for Visual Studio to avoid nvcc command line parsing issues
    set_property(TARGET example2_cuda PROPERTY CUDA_ARCHITECTURES "75")
else()
    set_property(TARGET example2_cuda PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()

# Example2 executable - link with CUDA library
add_executable(example2
    main.cpp
)

target_include_directories(example2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Link CUDA library and common library
target_link_libraries(example2 
    PUBLIC 
    common
    example2_cuda
)

# CUDA runtime is automatically linked when CUDA language is enabled

# MSVC target-specific compile options
if(MSVC)
    target_compile_options(example2 PRIVATE
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2>
    )
    target_link_options(example2 PRIVATE
        $<$<CONFIG:Debug>:/DEBUG>
    )
endif()

