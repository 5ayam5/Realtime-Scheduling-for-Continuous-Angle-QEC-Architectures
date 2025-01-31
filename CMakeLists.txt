CMAKE_MINIMUM_REQUIRED(VERSION 3.22)

project(RESCQSimulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

add_compile_options(-Wall -Wextra -pedantic -Ofast -g)

find_package(Boost REQUIRED)
include_directories( ${Boost_INCLUDE_DIRS} )

add_subdirectory(simulator)

add_executable(sim simulator/main.cpp)
target_link_libraries(sim main ${Boost_LIBRARIES})
