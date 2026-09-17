project(chp_source)

message(STATUS "The OpenBabel lib path is: ${OPENBABEL_LIB_PATH}")
# message(STATUS "The OpenBabel include dir is: ${OPENBABEL_INCLUDE_DIR}")
# message(STATUS "Include directorise have: ${INCLUDE_DIRECTORIES}")

add_executable(test_OBChemInfo testOBChemInfo.cpp ${CMAKE_SOURCE_DIR}/src/obcheminfo/mol.cpp)
target_link_libraries(test_OBChemInfo PRIVATE ${OPENBABEL_LIB_PATH})

add_executable(
    testUtils 
    testUtils.cpp 
    ${CMAKE_SOURCE_DIR}/src/utils/conformers.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/io.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/tools.cpp
)
target_link_libraries(testUtils PRIVATE ${OPENBABEL_LIB_PATH})
# target_include_directories(testUtils PUBLIC ${OPENBABEL_INCLUDE_DIR})


add_test(NAME TestOBChemInfo COMMAND test_OBChemInfo)
