# Define some utils function
function(highlight_message msg)  
    message(STATUS "\\033[1;32m${msg}\\033[0m")  # Green bold message  
endfunction()
highlight_message("This is a highlighted message!")

# Prefix setting
set(TORCHLIB_DIR "/home/zzy/software/libtorch")
set(OPENBABEL_LIB_PATH "/usr/local/lib/libopenbabel.so")
# Specify the path to the Eigen library  
set(EIGEN3_HEADER_DIR "${CMAKE_SOURCE_DIR}/third_party/eigen-3.4.0")

# set(OPENBABEL_LIB_DIR "/home/zzy/lib/openbabel/openbabel")
# set(nvtx3_dir "/home/zzy/software/NVTX/c/include")

# Optionally allow the user to specify the TORCHLIB_DIR at configure time  
# set(TORCHLIB_DIR "" CACHE PATH "Path to the PyTorch C++ library")  

# if(NOT TORCHLIB_DIR)  
#     # If not specified, check if the environment variable TORCHLIB_DIR is set  
#     if(DEFINED ENV{TORCHLIB_DIR})  
#         set(TORCHLIB_DIR $ENV{TORCHLIB_DIR})  
#         message(STATUS "Using TORCHLIB_DIR from environment variable: ${TORCHLIB_DIR}")  
#     endif()  
# endif() 

# find_package(OpenBabel3 REQUIRED)

message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")

# Specify output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

cmake_minimum_required(VERSION 3.5.0)
project(
    chp 
    VERSION 0.0.0 
    DESCRIPTION "a sub-package of hotpot"
    LANGUAGES C CXX
)

# Define your third-party library paths and include directories  
set(THIRD_PARTY_INCLUDE_DIRS  
    ${OPENBABEL_INCLUDE_DIR}
    ${EIGEN3_HEADER_DIR}
)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message("Third party include dirs are: ${THIRD_PARTY_INCLUDE_DIRS}")
include_directories(include)
include_directories(THIRD_PARTY_INCLUDE_DIRS)

# Configure and enable test
include(CPack)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

