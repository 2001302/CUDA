add_library(cuda STATIC
    add_vector.cu
)

if(MSVC)
    set_target_properties(cuda PROPERTIES
        CUDA_ARCHITECTURES "75"
    )
else()
    if(CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(cuda PROPERTIES
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        )
    endif()
endif()

target_include_directories(cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/include
)

if(MSVC)
    get_target_property(CUDA_COMPILE_OPTIONS cuda COMPILE_OPTIONS)
    if(CUDA_COMPILE_OPTIONS)
        list(REMOVE_ITEM CUDA_COMPILE_OPTIONS "/W4" "/permissive-")
        set_target_properties(cuda PROPERTIES COMPILE_OPTIONS "${CUDA_COMPILE_OPTIONS}")
    endif()
    target_compile_options(cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /wd4819>
    )
endif()

