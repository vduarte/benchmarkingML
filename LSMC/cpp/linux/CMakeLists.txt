cmake_minimum_required(VERSION 3.5)

#set(CMAKE_VERBOSE_MAKEFILE ON)

project(second_linux)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -O2 -Wall -pedantic -pedantic-errors")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CXX_STANDARD 11)

add_definitions(-DARMA_DONT_USE_WRAPPER -DARMA_USE_CXX11)

include_directories(../armadillo-9.400.4/include)
add_executable(second_project ../cpp_code.cpp)

# you may need to install blas and lapack first
# sudo apt-get libblas3 libblas-dev liblapack3 liblapack-dev
target_link_libraries(second_project blas lapack)


