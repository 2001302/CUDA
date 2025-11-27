# Algorithm library (C++ only)
add_library(algorithm_cpp STATIC
    algorithm.cpp
)

target_include_directories(algorithm_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(algorithm_cpp PUBLIC
    cuda
    common
)

# Python module (pybind11) - optional
find_package(Python COMPONENTS Interpreter Development QUIET)

if(Python_FOUND AND Python_Interpreter_FOUND)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../external/pybind11
                     ${CMAKE_CURRENT_BINARY_DIR}/pybind11)
    
    pybind11_add_module(algorithm
        python_binding.cpp
        algorithm.cpp
    )
    
    target_include_directories(algorithm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    
    target_link_libraries(algorithm PRIVATE
        cuda
        common
    )
    
    message(STATUS "Python bindings enabled")
else()
    message(STATUS "Python not found - skipping Python bindings")
endif()

if(MSVC)
    target_compile_options(algorithm_cpp PRIVATE
        $<$<CONFIG:Debug>:/Od /Zi>
        $<$<CONFIG:Release>:/O2>
    )
    target_link_options(algorithm_cpp PRIVATE
        $<$<CONFIG:Debug>:/DEBUG>
    )
    
    if(Python_FOUND AND Python_Interpreter_FOUND)
        target_compile_options(algorithm PRIVATE
            $<$<CONFIG:Debug>:/Od /Zi>
            $<$<CONFIG:Release>:/O2>
        )
        target_link_options(algorithm PRIVATE
            $<$<CONFIG:Debug>:/DEBUG>
        )
    endif()
endif()

