project(camera_calibration)
cmake_minimum_required(VERSION 2.8)

set(OUT_EXEC "camera_calibration")
file(GLOB SRC_FILES "./src/*.cpp")

set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(LIBCV_INCLUDE_DIR "/usr/local/include/cv" CACHE FILEPATH "Path to the libcv include directory")
set(LIBCV_LIB_DIR "/usr/local/lib" CACHE FILEPATH "Path to the libcv lib directory")

include_directories(${LIBCV_INCLUDE_DIR})
link_directories(${LIBCV_LIB_DIR})

add_executable(${OUT_EXEC} ${SRC_FILES})
target_link_libraries(${OUT_EXEC} "cv")


