add_executable(PlotTest main.cpp)
target_include_directories(PlotTest PUBLIC "../include")

# Include Python 2.7 for mathplotlib
find_package(PythonLibs 2.7)
target_include_directories(PlotTest PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(PlotTest ${PYTHON_LIBRARIES})
