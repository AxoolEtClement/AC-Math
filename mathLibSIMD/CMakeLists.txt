cmake_minimum_required(VERSION 3.25)
project(mathLibSIMD)

file(GLOB_RECURSE SIMD_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Si tu es sous MSVC et que tu as un fichier .asm spécifique
if(MSVC AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/dotProd.asm")
    enable_language(ASM_MASM)
    list(APPEND SIMD_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/dotProd.asm")
endif()

add_library(mathLibSIMD STATIC ${SIMD_SOURCES})
add_library(AC::mathLibSIMD ALIAS mathLibSIMD)

target_include_directories(mathLibSIMD
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(NOT MSVC)
    target_compile_options(mathLibSIMD PUBLIC
        -msse4.1
        -Wno-ignored-attributes   # std::array<__m128, N>
    )
endif()
