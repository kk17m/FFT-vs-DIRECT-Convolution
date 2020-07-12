#----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(fftConvolve)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11")

set(FFTW3_INCLUDE_PATH "**MISSING**" CACHE FILEPATH "description")
set(FFTW3_LIBRARIES "**MISSING**" CACHE FILEPATH "description")

message("\nFFTW3: \t" ${FFTW3_INCLUDE_PATH})

message("\nFFTW3: \t" ${FFTW3_LIBRARIES})



if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    if (NOT (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7))
        message(FATAL_ERROR "${PROJECT_NAME} requires g++ 4.7 or greater.")
    endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
else ()
    message(FATAL_ERROR "Your C++ compiler does not support C++11.")
endif ()

include_directories(${PROJECT_SOURCE_DIR}
                    ${FFTW3_INCLUDE_PATH})

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs
#
#file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
#file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

add_executable(${PROJECT_NAME} "fftConvolve.cc")

target_link_libraries(fftConvolve ${FFTW3_LIBRARIES})


#----------------------------------------------------------------------------
# Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
#
#install(TARGETS fftConvolve DESTINATION bin)
