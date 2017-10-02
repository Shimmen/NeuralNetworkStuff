add_executable(OjaUnsupervised main.cpp)
target_include_directories(OjaUnsupervised PUBLIC "/")
target_include_directories(OjaUnsupervised PUBLIC "../include")

set_property(TARGET OjaUnsupervised PROPERTY CXX_STANDARD 11)

# Include Python 2.7 for mathplotlib
find_package(PythonLibs 2.7)
target_include_directories(OjaUnsupervised PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(OjaUnsupervised ${PYTHON_LIBRARIES})
