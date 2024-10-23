cmake_minimum_required(VERSION 3.23)
message("Configuring Xyber Controller project")

project(xyber_controller LANGUAGES CXX C)

set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -Wl,--disable-new-dtags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--disable-new-dtags")

# Set target name
set(CUR_TARGET_NAME "example")

# Set file collection
file(GLOB_RECURSE src example/*.cpp)

# Add target
add_executable(${CUR_TARGET_NAME})

# Set source file of target
target_sources(${CUR_TARGET_NAME} PRIVATE ${src})

# Set include path of target
target_include_directories(
  ${CUR_TARGET_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Set link libraries of target
target_link_directories(${CUR_TARGET_NAME} PUBLIC lib)

target_link_libraries(${CUR_TARGET_NAME} PUBLIC xyber_controller)

install(
  TARGETS ${CUR_TARGET_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin COMPONENT ${CUR_TARGET_NAME}
)

install(
  DIRECTORY lib/
  DESTINATION bin
)
