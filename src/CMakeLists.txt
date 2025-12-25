find_package(Magnum REQUIRED GL Sdl2Application)

set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)


add_executable(MagnumGameOfLife GameOfLife.cpp ArcBall.cpp)

target_link_libraries(MagnumGameOfLife  PRIVATE 
  Magnum::GL 
  Magnum::Shaders 
  Magnum::Primitives 
  Magnum::MeshTools 
  Magnum::SceneGraph 
  Magnum::Application
  )

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA gefunden: ${CMAKE_CUDA_COMPILER}")
    set(USE_CUDA ON)
    set_target_properties(MagnumGameOfLife PROPERTIES CUDA_SEPERABLE_COMPILATION ON CUDA_STANDARD 14)
    set_property(TARGET MagnumGameOfLife PROPERTY CUDA_ARCHITECTURES 75)
    add_library(cuda_kernels STATIC GameOfLifeCUDA3d.cu)
    target_link_libraries(MagnumGameOfLife PRIVATE cuda_kernels)
else()
    message(STATUS "CUDA nicht gefunden – baue ohne CUDA")
    set(USE_CUDA OFF)
endif()

  # Make the executable a default target to build & run in Visual Studio
set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT MagnumGameOfLife)
