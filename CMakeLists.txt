cmake_minimum_required(VERSION 3.0.2)
set(OpenCV_DIR "/home/lovod/opencv-4.8.1/build")
project(tracking_kalman)

find_package(OpenCV 4.8.1 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf2_eigen
)


include_directories(
        include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(track src/track.cpp)
target_link_libraries(track ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_executable(kalman src/kalman.cpp)
target_link_libraries(kalman ${catkin_LIBRARIES} ${OpenCV_LIBS})
