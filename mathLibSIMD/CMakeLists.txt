# À renommer en mathLibSIMD/CMakeLists.txt

file(GLOB_RECURSE SIMD_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

if(MSVC)
    # Version MASM : uniquement sous MSVC
    enable_language(ASM_MASM)
    list(FILTER SIMD_SOURCES EXCLUDE REGEX "dotProd_intrinsics\\.cpp$")
    list(APPEND SIMD_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/dotProd.asm")
else()
    # GCC / Clang : le .asm (syntaxe MASM) n'est pas compilé,
    # on utilise la version intrinsics src/dotProd_intrinsics.cpp
    list(APPEND SIMD_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/dotProd_intrinsics.cpp")
    list(REMOVE_DUPLICATES SIMD_SOURCES)
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