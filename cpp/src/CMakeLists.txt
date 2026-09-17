project(chp_source)

# Specify the path to the Eigen library  
# set(EIGEN3_HEADER_DIR "${CMAKE_SOURCE_DIR}/lib/eigen-3.4.0/")
message(STATUS "The EIGEN3_HEADER_DIR is ${EIGEN3_HEADER_DIR}")
message(STATUS "The CMAKE prefix paths includes: ${CMAKE_PREFIX_PATH}")

if (DEFINED TORCHLIB_DIR)
    list(APPEND CMAKE_PREFIX_PATH ${TORCHLIB_DIR})
endif()

message(STATUS "The CMAKE_PREFIX_PATH is ${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
message(STATUS "Found torch libraries: ${TORCH_LIBRARIES}")

add_executable(chp main.cpp)
target_include_directories(chp PUBLIC ${EIGEN3_HEADER_DIR})
set_target_properties(chp PROPERTIES  
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin  
) 

target_link_libraries(chp "${TORCH_LIBRARIES}")
# set_property(TARGET chp PROPERTY CXX_STANDARD 17)

add_subdirectory(cheminfo)
