add_executable(example1
    main.cpp
)

target_link_libraries(example1 PUBLIC common)

target_include_directories(example1 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(MSVC)
    target_compile_options(example1 PRIVATE
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2>
    )
    target_link_options(example1 PRIVATE
        $<$<CONFIG:Debug>:/DEBUG>
    )
endif()

