
cmake_minimum_required(VERSION 3.5.1)

project(mcl)

set(OpenCV_DIR /usr/local/Cellar/opencv)
find_package(OpenCV 4 REQUIRED)

#set(CMAKE_PREFIX_PATH "/home/phi/libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu/libtorch/")
set(CMAKE_PREFIX_PATH "/opt/homebrew/Cellar/pytorch/2.2.0_8")
#set(CMAKE_PREFIX_PATH "/home/ros/libtorch")
find_package(Torch REQUIRED)

file(GLOB project_mcl *.cpp)

add_executable(main ${project_mcl})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

target_link_libraries(main ${OpenCV_LIBS} ${TORCH_LIBRARIES}) 

set_target_properties(main PROPERTIES
    CXX_STANDARD_REQUIRED ON
    CXX_STANDARD 17)

target_compile_options(main PRIVATE -Wall; -Wextra; -g)
