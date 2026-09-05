cmake_minimum_required(VERSION 3.16)
project(idr_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependency-free core library
add_library(idr_core
    src/engine.cpp
)

target_include_directories(idr_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(idr_core PRIVATE /W4)
else()
    target_compile_options(idr_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Unit tests
include(CTest)
if(BUILD_TESTING)
    find_package(Catch2 3 QUIET)

    if(NOT Catch2_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v3.5.3
            GIT_SHALLOW    TRUE
        )
        FetchContent_GetProperties(Catch2)
        if(NOT catch2_POPULATED)
            FetchContent_MakeAvailable(Catch2)
        endif()
    endif()

    add_executable(idr_tests tests/test_engine.cpp)
    target_link_libraries(idr_tests PRIVATE idr_core)

    if(TARGET Catch2::Catch2WithMain)
        target_link_libraries(idr_tests PRIVATE Catch2::Catch2WithMain)
    endif()

    if(MSVC)
        target_compile_options(idr_tests PRIVATE /W4)
    else()
        target_compile_options(idr_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME idr_core_tests COMMAND idr_tests)
endif()
