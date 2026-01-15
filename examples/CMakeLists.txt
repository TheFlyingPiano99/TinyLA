# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create test executable
add_executable(TinyLA_examples "examples.cpp")

# Link the library to the test executable
target_link_libraries(TinyLA_examples PRIVATE TinyLA::TinyLA)

# Cross product derivative test
add_executable(test_cross_deriv "../test_cross_standalone.cpp")
target_link_libraries(test_cross_deriv PRIVATE TinyLA::TinyLA)

# Rotation derivative test
add_executable(test_rotation_deriv "../test_rotation_derivative.cpp")
target_link_libraries(test_rotation_deriv PRIVATE TinyLA::TinyLA)
