add_executable(SupervisedLearning main.cpp neural-network.cpp)
target_include_directories(SupervisedLearning PUBLIC "/")
target_include_directories(SupervisedLearning PUBLIC "../include")

set_property(TARGET SupervisedLearning PROPERTY CXX_STANDARD 11)

# Include Python 2.7 for mathplotlib
find_package(PythonLibs 2.7)
target_include_directories(SupervisedLearning PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(SupervisedLearning ${PYTHON_LIBRARIES})
