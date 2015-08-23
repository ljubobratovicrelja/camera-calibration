project(camera_calibration)
cmake_minimum_required(VERSION 2.8)

set(OUT_EXEC "camera_calibration")
file(GLOB SRC_FILES "./src/*.cpp" "./src/*.c")

if (WIN32)
	set(CMAKE_CXX_FLAGS "/fopenmp /W3")
	set(CMAKE_CXX_FLAGS_DEBUG "/DEBUG")
	set(CMAKE_CXX_FLAGS_RELEASE "/Ox /DNDEBUG")

	set(LIBCV_INCLUDE_DIR "/usr/local/include/cv" CACHE PATH "Path to the libcv include directory")
	set(LIBCV_LIB "/usr/local/lib" CACHE FILEPATH "Path to the libcv lib file.")

else(WIN32)
	set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -Wall")
	set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

	set(LIBCV_INCLUDE_DIR "cv/include/" CACHE PATH "Path to the libcv include directory")
	set(LIBCV_LIB "cv.lib" CACHE FILEPATH "Path to the libcv lib file.")

endif(WIN32)

if (CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES DEBUG)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE MATCHES Release OR CMAKE_BUILD_TYPE MATCHES RELEASE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif(CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES DEBUG)

if(NOT (IS_DIRECTORY ${LIBCV_INCLUDE_DIR}))
	message( FATAL_ERROR "LibCV include directory is not valid." )
endif(NOT (IS_DIRECTORY ${LIBCV_INCLUDE_DIR}))

if(NOT (EXISTS  ${LIBCV_LIB}))
	message( FATAL_ERROR "LibCV lib file is not valid." )
endif(NOT (EXISTS ${LIBCV_LIB}))

include_directories(${LIBCV_INCLUDE_DIR})

add_executable(${OUT_EXEC} ${SRC_FILES})
target_link_libraries(${OUT_EXEC} ${LIBCV_LIB})


