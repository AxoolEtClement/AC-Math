cmake_minimum_required(VERSION 3.20)

project(BasicMathsLib VERSION 1.0.0 LANGUAGES CXX)


# Standard C++23 requis
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export des commandes de compilation pour LSP / VS Code (compile_commands.json)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Detection des fichiers sources et headers
file(GLOB_RECURSE LIB_HEADERS "include/*.h" "include/*.hpp")
file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")
file(GLOB_RECURSE TEST_SOURCES "Tests/*.cpp")
file(GLOB_RECURSE TEST_HEADERS "Tests/*.h" "Tests/*.hpp")

# Inclusion des dossiers de têtes
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/Tests
)

# Configuration de la bibliothèque (Header-Only si aucun .cpp dans src/)
if(LIB_SOURCES)
    add_library(${PROJECT_NAME} STATIC ${LIB_SOURCES} ${LIB_HEADERS})
    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
else()
    add_library(${PROJECT_NAME} INTERFACE)
    target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    message(STATUS "Bibliothèque configurée en mode Header-Only")
endif()

# Flags de compilation selon le compilateur
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Cible pour les tests automatisés
if(TEST_SOURCES)
    add_executable(MathTests ${TEST_SOURCES} ${TEST_HEADERS})
    
    if(TARGET ${PROJECT_NAME})
        if(LIB_SOURCES)
            target_link_libraries(MathTests PRIVATE ${PROJECT_NAME})
        else()
            target_link_libraries(MathTests PRIVATE ${PROJECT_NAME})
        endif()
    endif()

    # Définition pour basculer le test runner sous Linux/Clang
    target_compile_definitions(MathTests PRIVATE COMPAT_TEST_RUNNER)

    message(STATUS "Executable de test 'MathTests' configuré.")
endif()
