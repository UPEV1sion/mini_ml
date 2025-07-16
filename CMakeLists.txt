cmake_minimum_required(VERSION 3.31)
project(ai C)

set(CMAKE_C_STANDARD 11)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(ai
        src/main.c
        src/regression/gradients.c
        src/ml/ml.c
        src/classification/gradients.c
)

target_link_libraries(ai m)