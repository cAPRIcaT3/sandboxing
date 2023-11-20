cmake_minimum_required(VERSION 3.15)
project(YourProjectName)

# Specify the project source directory and binary directory
set(CMAKE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(CMAKE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Find Python
find_package(Python REQUIRED)

# Set the Python script file path
set(PYTHON_SCRIPT_PATH "${CMAKE_SOURCE_DIR}/src/generate_comment.py")

# Add a custom target for running the Python script
add_custom_target(run_python_script
    COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_SCRIPT_PATH}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ${PYTHON_SCRIPT_PATH}
)

# Define additional dependencies or options as needed
# For example, adding input files as dependencies for the script
file(GLOB INPUT_FILES "${CMAKE_SOURCE_DIR}/src/files/*")
add_custom_target(copy_inputs
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/files ${CMAKE_CURRENT_BINARY_DIR}/input_files
    DEPENDS ${INPUT_FILES}
)
add_dependencies(run_python_script copy_inputs)

# Add a custom target for copying the generated output files to a specific location
add_custom_target(copy_outputs
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/generated_outputs ${CMAKE_INSTALL_PREFIX}/share/YourProjectName
    DEPENDS ${PYTHON_SCRIPT_PATH} ${INPUT_FILES}
)
