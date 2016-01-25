CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

PROJECT("testsAll")

ADD_SUBDIRECTORY("vector/")
ADD_SUBDIRECTORY("forceMatrixKernel/")
ADD_SUBDIRECTORY("updatePositionsKernel/")
ADD_SUBDIRECTORY("simulationClass/")
ADD_SUBDIRECTORY("simulationTest/")
ADD_SUBDIRECTORY("benchmark/")
