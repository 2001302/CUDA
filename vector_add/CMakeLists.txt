add_executable(vector_add
    vector_add.cu
)

target_include_directories(vector_add PRIVATE
    ${CMAKE_SOURCE_DIR}/common
)

# CUDA 컴파일 옵션
target_compile_options(vector_add PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)


