project(chp_source)

add_library(
    utils.so
    SHARED
    io.cpp
    tools.cpp
    conformers.cpp
)