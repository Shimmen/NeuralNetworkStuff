add_executable(CombinedNetworks main.cpp neural-network.cpp)
target_include_directories(CombinedNetworks PUBLIC "/")
target_include_directories(CombinedNetworks PUBLIC "../include")

set_property(TARGET CombinedNetworks PROPERTY CXX_STANDARD 11)

# Include Python 2.7 for mathplotlib
find_package(PythonLibs 2.7)
target_include_directories(CombinedNetworks PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(CombinedNetworks ${PYTHON_LIBRARIES})
