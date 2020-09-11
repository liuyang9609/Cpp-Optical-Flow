cmake_minimum_required(VERSION 2.8) 
PROJECT (hscpp)
find_package(OpenCV REQUIRED )
set( NAME_SRC
    src/main.cpp 
    src/hornSchunck.cpp
    Src/plotFlow.cpp
)


INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( hscpp ${NAME_SRC} )

target_link_libraries( hscpp ${OpenCV_LIBS} )         

