cmake_minimum_required( VERSION 3.5.1 )
project( sgm )

message( "** Using debug mode." )
add_definitions(-DDEBUG)
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=gnu++0x" )

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})

add_library(sgm SHARED sgm.cpp)
