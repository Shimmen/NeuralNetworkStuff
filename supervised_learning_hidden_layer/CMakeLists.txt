add_executable(SupervisedLearningHiddenLayer main.cpp neural-network.cpp)
target_include_directories(SupervisedLearningHiddenLayer PUBLIC "/")
target_include_directories(SupervisedLearningHiddenLayer PUBLIC "../include")

set_property(TARGET SupervisedLearningHiddenLayer PROPERTY CXX_STANDARD 11)

# Include Python 2.7 for mathplotlib
find_package(PythonLibs 2.7)
target_include_directories(SupervisedLearningHiddenLayer PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(SupervisedLearningHiddenLayer ${PYTHON_LIBRARIES})
