cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(readmnist)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


set(CMAKE_PREFIX_PATH /media/work/dependent_libs/libtorch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(readmnist 
                src/main.cpp
                src/ReadMnistData.cpp
                include/ReadMnistData.h
                include/MnistMLP.h
              )

add_executable(testmnist src/test_model.cpp)

target_include_directories(readmnist PRIVATE include)

target_include_directories(testmnist PRIVATE include)


target_link_libraries(readmnist "${TORCH_LIBRARIES}")
target_link_libraries(readmnist "${OpenCV_LIBS}" "${GLib_LIBS}")

target_link_libraries(testmnist "${TORCH_LIBRARIES}")
target_link_libraries(testmnist "${OpenCV_LIBS}" "${GLib_LIBS}")


set_property(TARGET readmnist PROPERTY CXX_STANDARD 17)

set_property(TARGET testmnist PROPERTY CXX_STANDARD 17)

