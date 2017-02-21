cmake_minimum_required(VERSION 3.0)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMCOnfig.cmake in: ${LLVM_DIR}")

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

# set(LLVM_LINK_COMPONENTS support)

# add_clang_executable(loop-convert LoopConvert.cpp)


# set(HELIUM_SOURCES
#   helium.cc
#   hetok.cc)

# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# output target to bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(loop-convert LoopConvert.cpp)
# target_link_libraries(hetok helium)

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs support core irreader)

# Link against LLVM libraries
target_link_libraries(loop-convert ${llvm_libs})

# target_link_libraries(loop-convert
#   clangTooling
#   # clangBasic
#   # clangASTMatchers
#   )
