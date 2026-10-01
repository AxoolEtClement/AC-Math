cmake_minimum_required(VERSION 3.25)

project(mathLibCPP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. Recherche récursive des fichiers d'en-tête
file(GLOB_RECURSE HEADERS
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)

# 2. Recherche récursive des fichiers sources (.cpp)
file(GLOB_RECURSE SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# 3. Bibliothèque header-only (INTERFACE)
add_library(mathLibCPP INTERFACE)
target_sources(mathLibCPP INTERFACE ${HEADERS})
target_include_directories(mathLibCPP INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

enable_testing()
add_subdirectory(mathLibSIMD)
add_subdirectory(tests)
add_subdirectory(benchmark)


target_compile_options(mathLibSIMD PUBLIC -msse4.1)

# Avertissements stricts
if(MSVC)
    target_compile_options(mathLibCPP INTERFACE /W4 /permissive- /Zc:__cplusplus)
else()
    target_compile_options(mathLibCPP INTERFACE -Wall -Wextra -Wpedantic)
endif()



# 4. Création du programme exécutable rattaché aux fichiers .cpp
add_executable(app ${SOURCES})

# 5. Liaison de l'exécutable avec la bibliothèque de maths
target_link_libraries(app PRIVATE mathLibCPP)

