add_executable(DeterministicHopfieldModel main.cpp)
target_include_directories(DeterministicHopfieldModel PUBLIC "../include")

set_property(TARGET DeterministicHopfieldModel PROPERTY CXX_STANDARD 11)

# Include Python 2.7 for mathplotlib
find_package(PythonLibs 2.7)
target_include_directories(DeterministicHopfieldModel PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(DeterministicHopfieldModel ${PYTHON_LIBRARIES})
