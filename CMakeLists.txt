cmake_minimum_required(VERSION 3.10)

if(COMPILE_WITH_CUDA)
    project(TinyLAProject VERSION 0.0.2 LANGUAGES CXX CUDA)
else()
    project(TinyLAProject VERSION 0.0.2 LANGUAGES CXX)
endif()

add_subdirectory(include)
if(COMPILE_TESTS)
    add_subdirectory(tests)
    add_subdirectory(examples)
endif()
