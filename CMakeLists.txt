cmake_minimum_required(VERSION 3.20)
project(llama-cpp-python)

add_executable(generate_comment src/generate_comment.py)

target_link_libraries(generate_comment llama-cpp-python)

install(TARGETS generate_comment DESTINATION bin)
