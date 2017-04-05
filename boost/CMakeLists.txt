cmake_minimum_required(VERSION 3.0)
find_package( Boost 1.40 COMPONENTS system filesystem program_options regex timer REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )
link_libraries(
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_LIBRARIES})

add_executable(file filesystem.cpp)

