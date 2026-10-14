cmake_minimum_required(VERSION 3.25)

project(mathLibCPP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# 1. Recherche récursive des fichiers d'en-tête (pour l'affichage dans les IDE)
file(GLOB_RECURSE HEADERS
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)

# 2. Bibliothèque header-only (INTERFACE)
add_library(mathLibCPP INTERFACE)
add_library(AC::mathLibCPP ALIAS mathLibCPP)

target_sources(mathLibCPP INTERFACE ${HEADERS})
target_include_directories(mathLibCPP INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# 3. Bibliothèque SIMD (cible mathLibSIMD, définie dans mathLibSIMD/CMakeLists.txt)
add_subdirectory(mathLibSIMD)

# Lier mathLibCPP suffit désormais à récupérer mathLibSIMD et ses includes
target_link_libraries(mathLibCPP INTERFACE mathLibSIMD)

# 4. Avertissements stricts
if(MSVC)
    target_compile_options(mathLibCPP INTERFACE /W4 /permissive- /Zc:__cplusplus)
else()
    target_compile_options(mathLibCPP INTERFACE -Wall -Wextra -Wpedantic)
endif()

# 5. Tests et benchmarks : uniquement quand la lib est le projet principal,
#    pas quand elle est tirée par un autre projet (ex. AC-CPUEngine)
if(PROJECT_IS_TOP_LEVEL)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(benchmark)
endif()