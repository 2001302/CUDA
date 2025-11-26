add_executable(example2
    main.cpp
)

target_include_directories(example2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/include
)

target_link_libraries(example2 PUBLIC
    common
    cuda
)

if(MSVC)
    target_compile_options(example2 PRIVATE
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2>
    )
    target_link_options(example2 PRIVATE
        $<$<CONFIG:Debug>:/DEBUG>
    )
endif()

