cmake_minimum_required(VERSION 3.13)

find_package(GTest REQUIRED)

message("GTEST_INCLUDE_DIRS = ${GTEST_INCLUDE_DIRS}")

add_executable(myEx main.cpp)
target_link_libraries(myEx ${GTEST_LIBRARIES} pthread)
