# Example2 executable - link with CUDA common library and common library
add_executable(example2
    main.cpp
)

target_include_directories(example2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Link CUDA common library and common library
target_link_libraries(example2 
    PUBLIC 
    common
    cuda
)

# MSVC target-specific compile options
if(MSVC)
    target_compile_options(example2 PRIVATE
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2>
    )
    target_link_options(example2 PRIVATE
        $<$<CONFIG:Debug>:/DEBUG>
        # Suppress LNK4098 warning by excluding static runtime library
        /NODEFAULTLIB:LIBCMT
    )
endif()

