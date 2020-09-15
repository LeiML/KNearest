cmake_minimum_required(VERSION 3.17)
project(KNearest)

set(CMAKE_CXX_STANDARD 20)
include_directories("D://ProgramFiles//Matrix//include")
link_directories("D://ProgramFiles//Matrix//lib")

add_executable(KNearest main.cpp KNearest.hpp KNearest.cpp)
target_link_libraries(KNearest Matrix)