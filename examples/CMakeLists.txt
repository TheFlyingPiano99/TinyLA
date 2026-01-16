# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create examples executable
add_executable(TinyLA_examples "examples.cpp")

# Link the library to the examples executable
target_link_libraries(TinyLA_examples PRIVATE TinyLA::TinyLA)

