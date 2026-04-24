cmake_policy(SET CMP0048 NEW)
project("DSP-SLAM")
cmake_minimum_required(VERSION 3.2)

# c++ version
set(CMAKE_CXX_STANDARD 17)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_LESS "3.2")
    message(FATAL_ERROR "OpenCV >= 3.2 required. Found: ${OpenCV_VERSION}")
endif()
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

# OpenCV 4.x removed some old C-style constants. Provide backward-compatibility defines.
if(OpenCV_VERSION VERSION_GREATER_EQUAL "4.0.0")
    add_compile_definitions(
        CV_LOAD_IMAGE_UNCHANGED=-1
        CV_RGB2GRAY=cv::COLOR_RGB2GRAY
        CV_BGR2GRAY=cv::COLOR_BGR2GRAY
        CV_RGBA2GRAY=cv::COLOR_RGBA2GRAY
        CV_BGRA2GRAY=cv::COLOR_BGRA2GRAY
    )
endif()

find_package(Eigen3 3.4.0 REQUIRED)
find_package(Pangolin REQUIRED)

# Pangolin was built with GCC 9+ and its pango_video target exports
# -Wno-error=deprecated-copy and -Wno-error=pessimizing-move as
# INTERFACE_COMPILE_OPTIONS. GCC 8 doesn't know these warnings at all,
# so passing them causes compilation errors. Clear the options.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    if(TARGET pango_video)
        set_target_properties(pango_video PROPERTIES INTERFACE_COMPILE_OPTIONS "")
    endif()
endif()

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
)

# You need to set to your own Python paths
add_subdirectory(pybind11)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED
        src/System.cc
        src/System_util.cc
        src/Tracking.cc
        src/Tracking_util.cc
        src/LocalMapping.cc
        src/LocalMapping_util.cc
        src/LoopClosing.cc
        src/LoopClosing_util.cc
        src/ORBextractor.cc
        src/ORBmatcher.cc
        src/FrameDrawer.cc
        src/Converter.cc
        src/MapPoint.cc
        src/MapPoint_util.cc
        src/KeyFrame.cc
        src/KeyFrame_util.cc
        src/Map.cc
        src/Map_util.cc
        src/MapDrawer.cc
        src/Optimizer.cc
        src/Optimizer_util.cc
        src/PnPsolver.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Initializer.cc
        src/Viewer.cc
        src/ObjectDetection.cc
        src/MapObject.cc
        src/ObjectDrawer.cc
        src/ObjectRenderer.cc
        )
target_link_libraries(${PROJECT_NAME}
        pybind11::embed
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
add_executable(dsp_slam dsp_slam.cc)
target_link_libraries(dsp_slam ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
add_executable(dsp_slam_mono dsp_slam_mono.cc)
target_link_libraries(dsp_slam_mono ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
add_executable(dsp_slam_rgbd dsp_slam_rgbd.cc)
target_link_libraries(dsp_slam_rgbd ${PROJECT_NAME})