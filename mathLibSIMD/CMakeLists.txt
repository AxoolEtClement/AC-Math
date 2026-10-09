cmake_minimum_required(VERSION 3.25)
project(mathLibSIMD LANGUAGES CXX ASM_MASM)

set(ASM_NAME "${CMAKE_CURRENT_SOURCE_DIR}/src/dotProd.asm")

# Ensure CMake treats .asm files as assembly source files for MASM
set_source_files_properties(
    ${ASM_NAME}
    PROPERTIES LANGUAGE ASM_MASM
)

file(GLOB_RECURSE SOURCES
    CONFNIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    ${ASM_NAME}
)

file(GLOB_RECURSE HEADERS
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)

add_library(mathLibSIMD ${SOURCES} ${HEADERS})

target_include_directories(mathLibSIMD PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
