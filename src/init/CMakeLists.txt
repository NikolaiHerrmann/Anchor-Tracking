cmake_minimum_required(VERSION 2.8)                            
PROJECT(tracker)                                         

FIND_PACKAGE( OpenCV REQUIRED )                              
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

file(GLOB TRACKER_SRC
     "*/*.h"
     "*/*.cpp"
     "*.h"
     "*.cpp"
)

ADD_EXECUTABLE(${PROJECT_NAME} ${TRACKER_SRC})                         
TARGET_LINK_LIBRARIES (${PROJECT_NAME} ${OpenCV_LIBS})