# ---------------------------------------------------------------
# $Revision: 1.1 $
# $Date: 2013-02-14 $
# ---------------------------------------------------------------
# Programmer:  Eddy Banks @ LLNL
# ---------------------------------------------------------------
# Copyright (c) 2013, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see the LICENSE file.
# ---------------------------------------------------------------
# SuperLU tests for SUNDIALS CMake-based configuration.
#    - loosely based on SundialsLapack.cmake
# 

SET(SUPERLU_FOUND FALSE)

# set SUPERLU_LIBRARIES
include(FindSUPERLU)
# If we have the SUPERLU libraries, test them
if(SUPERLU_LIBRARIES)
  PRINT_WARNING("SundialsSuperLU.cmake SUPERLU_LIBRARIES" "${SUPERLU_LIBRARIES}")
  message(STATUS "Looking for SUPERLU libraries... OK")
  # Create the SuperLUTest directory
  set(SuperLUTest_DIR ${PROJECT_BINARY_DIR}/SuperLUTest)
  file(MAKE_DIRECTORY ${SuperLUTest_DIR})
  # Create a CMakeLists.txt file 
  file(WRITE ${SuperLUTest_DIR}/CMakeLists.txt
    "PROJECT(ltest C)\n"
    "SET(CMAKE_VERBOSE_MAKEFILE ON)\n"
    "SET(CMAKE_BUILD_TYPE \"${CMAKE_BUILD_TYPE}\")\n"
    "SET(CMAKE_C_FLAGS \"${CMAKE_C_FLAGS}\")\n"
    "SET(CMAKE_C_FLAGS_RELEASE \"${CMAKE_C_FLAGS_RELEASE}\")\n"
    "SET(CMAKE_C_FLAGS_DEBUG \"${CMAKE_C_FLAGS_DEBUG}\")\n"
    "SET(CMAKE_C_FLAGS_RELWITHDEBUGINFO \"${CMAKE_C_FLAGS_RELWITHDEBUGINFO}\")\n"
    "SET(CMAKE_C_FLAGS_MINSIZE \"${CMAKE_C_FLAGS_MINSIZE}\")\n"
    "ADD_EXECUTABLE(ltest ltest.c)\n"
    "TARGET_LINK_LIBRARIES(ltest ${SUPERLU_LIBRARIES})\n")    
# TODO: Eddy - fix this test
# Create a C source file which calls a SuperLU function
  file(WRITE ${SuperLUTest_DIR}/ltest.c
    "int main(){\n"
    "int n=1;\n"
    "double x, y;\n"
    "return(0);\n"
    "}\n")
  # Attempt to link the "ltest" executable
  try_compile(LTEST_OK ${SuperLUTest_DIR} ${SuperLUTest_DIR} ltest OUTPUT_VARIABLE MY_OUTPUT)
      
  # To ensure we do not use stuff from the previous attempts, 
  # we must remove the CMakeFiles directory.
  file(REMOVE_RECURSE ${SuperLUTest_DIR}/CMakeFiles)
  # Process test result
  if(LTEST_OK)
    message(STATUS "Checking if SuperLU works... OK")
    set(SUPERLU_FOUND TRUE)
  else(LTEST_OK)
    message(STATUS "Checking if SuperLU works... FAILED")
  endif(LTEST_OK)
else(SUPERLU_LIBRARIES)
  message(STATUS "Looking for SUPERLU libraries... FAILED")
endif(SUPERLU_LIBRARIES)
