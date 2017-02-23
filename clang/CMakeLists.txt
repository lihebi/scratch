cmake_minimum_required(VERSION 3.0)

## Directory setup
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# output target to bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

## Setting thread
# this is to add -pthread to the option, for gtest
find_package (Threads)
link_libraries(${CMAKE_THREAD_LIBS_INIT})

## Setting LLVM
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMCOnfig.cmake in: ${LLVM_DIR}")
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
set(LLVM_LINK_COMPONENTS support)

## Setting Clang
# this is required
find_package(Clang REQUIRED CONFIG)

# why I need this??? Otherwise error: undefined reference to typeinfo for xxx
add_compile_options(-fno-rtti)

# link these by default
link_libraries(clang clangTooling clangFrontend clangFrontendTool)

add_executable(ast ast.cpp)
# target_link_libraries(ast clang clangTooling clangFrontend clangFrontendTool)

add_executable(token token.cpp)
# target_link_libraries(token clang clangTooling clangFrontend clangFrontendTool)

add_executable(rewriter rewriter.cpp)
