include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 2. Recherche récursive des fichiers sources (.cpp)
file(GLOB_RECURSE SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Crée l'exécutable de test
add_executable(run_tests ${SOURCES})

target_include_directories(run_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Utilisation du bon nom de cible : mathLibCPP
target_link_libraries(run_tests PRIVATE mathLibCPP gtest_main)

include(GoogleTest)
gtest_discover_tests(run_tests)