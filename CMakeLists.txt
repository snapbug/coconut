cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(coconut)

# Allow avx instructions, and target native architecture
add_definitions(-mavx -march=native)

# findpackage(blaze) does not work
include("${CMAKE_CURRENT_LIST_DIR}/external/blaze/build/cmake/blaze-targets.cmake")
find_package(avro REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)

#
# This target doesn't do anything, but anything that depends on this will always be rebuilt.
#
add_custom_command(
  OUTPUT always_rebuild
  COMMAND cmake -E echo
  )

#
# Make the generator program
#
add_executable(coconut-gen
  coconut-gen.cpp
  )
target_link_libraries(coconut-gen
  avrocpp
  )

#
# Use the generator to make the main, and make that
#
add_custom_command(
  OUTPUT coconut-server.cpp
  COMMAND coconut-gen "${CMAKE_CURRENT_LIST_DIR}/weights.avro"
  always_rebuild
  )

add_executable(coconut-server
  coconut-server.cpp
  gen-cpp/QuestionAnswering.h
  gen-cpp/QuestionAnswering.cpp
  gen-cpp/qa_constants.cpp
  gen-cpp/qa_types.cpp
  )
target_link_libraries(coconut-server
  blaze::blaze
  thrift
)

add_executable(coconut-client
  coconut-client.cpp
  gen-cpp/QuestionAnswering.h
  gen-cpp/QuestionAnswering.cpp
  gen-cpp/qa_constants.cpp
  gen-cpp/qa_types.cpp
  )
target_link_libraries(coconut-client
  thrift
  )
