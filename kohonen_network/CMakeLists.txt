add_executable(KohonenNetwork main.cpp kohonen-network.cpp)
target_include_directories(KohonenNetwork PUBLIC "/")
target_include_directories(KohonenNetwork PUBLIC "../include")

set_property(TARGET KohonenNetwork PROPERTY CXX_STANDARD 11)

# Include Python 2.7 for mathplotlib
find_package(PythonLibs 2.7)
target_include_directories(KohonenNetwork PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(KohonenNetwork ${PYTHON_LIBRARIES})
