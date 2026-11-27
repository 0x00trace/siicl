# Builds libsiicl_dynarmic.so, the C interface to dynarmic that siicl.dynarmic loads.
#
#   git submodule update --init native/dynarmic
#   cmake -S native -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/native
#
# dynarmic's own libraries come with its source (native/dynarmic/externals); Boost comes from
# the system (pacman package boost).

cmake_minimum_required(VERSION 3.16)
project(siicl_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Some of dynarmic's bundled libraries declare a CMake version older than CMake 4 accepts.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(DYNARMIC_FRONTENDS "A64" CACHE STRING "" FORCE)
set(DYNARMIC_TESTS OFF CACHE BOOL "" FORCE)
set(DYNARMIC_USE_BUNDLED_EXTERNALS ON CACHE BOOL "" FORCE)
set(DYNARMIC_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
set(DYNARMIC_USE_PRECOMPILED_HEADERS OFF CACHE BOOL "" FORCE)
add_subdirectory(dynarmic EXCLUDE_FROM_ALL)

add_library(siicl_dynarmic SHARED siicl_dynarmic.cpp)
target_link_libraries(siicl_dynarmic PRIVATE dynarmic atomic)
set_target_properties(siicl_dynarmic PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
