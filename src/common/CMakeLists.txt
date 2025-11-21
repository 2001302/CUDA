# Common library (shared utilities)
add_library(common STATIC
    logger.cpp
)

target_include_directories(common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
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

