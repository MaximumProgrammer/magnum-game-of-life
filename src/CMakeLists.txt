find_package(Magnum REQUIRED GL Sdl2Application)

set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

add_executable(GameOfLife GameOfLife.cpp ArcBall.cpp)
target_link_libraries(GameOfLife PRIVATE 
  Magnum::GL 
  Magnum::Shaders 
  Magnum::Primitives 
  Magnum::MeshTools 
  Magnum::SceneGraph 
  Magnum::Application
  )

  # Make the executable a default target to build & run in Visual Studio
set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT GameOfLife)
