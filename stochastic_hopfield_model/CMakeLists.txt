add_executable(StochasticHopfieldModel main.cpp)
target_include_directories(StochasticHopfieldModel PUBLIC "../include")

set_property(TARGET StochasticHopfieldModel PROPERTY CXX_STANDARD 11)

# Include Python 2.7 for mathplotlib
find_package(PythonLibs 2.7)
target_include_directories(StochasticHopfieldModel PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(StochasticHopfieldModel ${PYTHON_LIBRARIES})
