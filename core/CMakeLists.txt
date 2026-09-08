cmake_minimum_required(VERSION 3.16)
project(idr_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependency-free core library
add_library(idr_core
    src/engine.cpp
    src/strapdown.cpp
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

# Replay driver CLI
add_executable(idr_replay
    src/replay/replay_driver.cpp
)
target_link_libraries(idr_replay PRIVATE idr_core)

if(MSVC)
    target_compile_options(idr_replay PRIVATE /W4)
else()
    target_compile_options(idr_replay PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Unit tests
include(CTest)
if(BUILD_TESTING)
    option(IDR_USE_FETCHED_GTEST "Fetch and link GoogleTest" ON)

    # Legacy MinGW 32-bit compilers (< GCC 8) lack C++11 std::mutex / threading support.
    # On those toolchains, use the zero-dependency test runner so builds succeed locally.
    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.0")
        set(IDR_USE_FETCHED_GTEST OFF)
    endif()

    add_executable(idr_tests tests/test_engine.cpp)
    target_link_libraries(idr_tests PRIVATE idr_core)

    add_executable(idr_strapdown_tests tests/test_strapdown.cpp)
    target_link_libraries(idr_strapdown_tests PRIVATE idr_core)

    if(IDR_USE_FETCHED_GTEST)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
            GIT_SHALLOW    TRUE
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)

        target_compile_definitions(idr_tests PRIVATE IDR_USE_GTEST=1)
        target_link_libraries(idr_tests PRIVATE GTest::gtest_main)

        include(GoogleTest)
        gtest_discover_tests(idr_tests)
        add_test(NAME idr_strapdown_tests COMMAND idr_strapdown_tests)
    else()
        add_test(NAME idr_core_tests COMMAND idr_tests)
        add_test(NAME idr_strapdown_tests COMMAND idr_strapdown_tests)
    endif()

    if(MSVC)
        target_compile_options(idr_tests PRIVATE /W4)
        target_compile_options(idr_strapdown_tests PRIVATE /W4)
    else()
        target_compile_options(idr_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(idr_strapdown_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()
