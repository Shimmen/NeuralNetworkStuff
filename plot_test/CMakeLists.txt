add_executable(PlotTest main.cpp)
target_include_directories(PlotTest PUBLIC "../include")

set_property(TARGET PlotTest PROPERTY CXX_STANDARD 11)

# Include Python 2.7 for mathplotlib
find_package(PythonLibs 2.7)
target_include_directories(PlotTest PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(PlotTest ${PYTHON_LIBRARIES})
