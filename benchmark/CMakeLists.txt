include(FetchContent)

FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.1.0
    GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(nanobench)


add_executable(benchmark benchmark.cpp)

target_link_libraries(benchmark PRIVATE nanobench)
target_link_libraries(benchmark PRIVATE mathLibCPP)
target_link_libraries(benchmark PRIVATE mathLibSIMD)
