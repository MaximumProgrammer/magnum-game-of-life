cmake_minimum_required(VERSION 3.5...3.20)

project(GameOfLife)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA gefunden: ${CMAKE_CUDA_COMPILER}")
    set(USE_CUDA ON)
    #set_target_properties(MagnumGameOfLife PROPERTIES CUDA_SEPERABLE_COMPILATION ON CUDA_STANDARD 14)
    #set_property(TARGET MagnumGameOfLife PROPERTY CUDA_ARCHITECTURES 75)
else()
    message(STATUS "CUDA nicht gefunden – baue ohne CUDA")
    set(USE_CUDA OFF)
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules/" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

# Add Corrade as a subproject
add_subdirectory(corrade EXCLUDE_FROM_ALL)

# Add Magnum as a subproject, enable Sdl2Application
set(MAGNUM_WITH_SDL2APPLICATION ON CACHE BOOL "" FORCE)
add_subdirectory(magnum EXCLUDE_FROM_ALL)

add_subdirectory(src)
