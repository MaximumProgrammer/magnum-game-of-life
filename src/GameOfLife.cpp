/*
    This file is part of Magnum.

    Original authors — credit is appreciated but not required:

        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
        2020, 2021, 2022, 2023, 2024, 2025
             — Vladimír Vondruš <mosra@centrum.cz>

    This is free and unencumbered software released into the public domain.

    Anyone is free to copy, modify, publish, use, compile, sell, or distribute
    this software, either in source code form or as a compiled binary, for any
    purpose, commercial or non-commercial, and by any means.

    In jurisdictions that recognize copyright laws, the author or authors of
    this software dedicate any and all copyright interest in the software to
    the public domain. We make this dedication for the benefit of the public
    at large and to the detriment of our heirs and successors. We intend this
    dedication to be an overt act of relinquishment in perpetuity of all
    present and future rights to this software under copyright law.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Shaders/PhongGL.h>
#include <Magnum/Trade/MeshData.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "backend-game-of-life.h"

namespace Magnum {
namespace Project {

using namespace Math::Literals;

class GameOfLife : public Platform::Application {
  public:
      explicit GameOfLife(const Arguments& arguments);
      ~GameOfLife();

  private:
  void drawEvent() override;
  void pointerReleaseEvent(PointerEvent& event) override;
  void pointerMoveEvent(PointerMoveEvent& event) override;
  void scrollEvent(ScrollEvent& event) override;
  void keyPressEvent(KeyEvent& event) override;
  void update();

  PlayGround3D<Field3D<int>> _playground;
  std::vector<GL::Mesh> _meshs;

  Shaders::PhongGL _shader;

  Matrix4 _transformation, _projection;
  Color3 _color;
  float _offset = 0;
  float _rotate = 0;
  bool _running = true;
  std::thread _thread;
};

GameOfLife::GameOfLife(const Arguments& arguments)
    : Platform::Application{arguments, Configuration{}.setTitle(
                                           "Magnum Primitives Example")},
      _playground(makePlayGround(PlayDefinition3D{10, 10, 10}))
{
  GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
  GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

  const auto width = _playground.getPlayGround().getWidth();
  const auto heigth = _playground.getPlayGround().getHeigth();
  const auto deep = _playground.getPlayGround().getDeep();

  std::cout << this->windowSize().x() << " " << this->windowSize().y() << std::endl;

  _projection =
      Matrix4::perspectiveProjection(
          35.0_degf, Vector2{windowSize()}.aspectRatio(), 0.001f, 250.0f) *
      Matrix4::translation({-25.0f, -25.0f, -250.0f});

  _color = Color3::fromHsv({35.0_degf, 1.0f, 1.0f});

  for (size_t i = 0; i < width * heigth * deep; ++i) {
    auto mesh = MeshTools::compile(Primitives::cubeSolid());
    _meshs.push_back(std::move(mesh));
  }

  _thread = std::thread(&GameOfLife::update, this);
}

GameOfLife::~GameOfLife()
{
  _running = false;
  while (!_thread.joinable()) {
  };
  _thread.join();
}

void GameOfLife::drawEvent()
{
  GL::defaultFramebuffer.clear(GL::FramebufferClear::Color |
                               GL::FramebufferClear::Depth);

  auto projection = _projection + Matrix4::translation({0.0f, 0.0f, _offset});

  const auto width = _playground.getPlayGround().getWidth();
  const auto heigth = _playground.getPlayGround().getHeigth();
  const auto deep = _playground.getPlayGround().getDeep();

  for (size_t i = 0; i < width; ++i) {
    for (size_t j = 0; j < heigth; ++j) {
      for (size_t k = 0; k < deep; ++k) {
        if (const auto isAlive = _playground.getValue({i, j, k}); isAlive) {
          auto _transformation = Matrix4::translation(
              {-25 + 10.0f * i, -25 + 10.0f * j, -25 + 10.0f * k});

          const auto pos = i + width * j + (width + heigth) * k;

          _shader.setLightPositions({{1.4f, 1.0f, 0.75f, 0.0f}})
              .setDiffuseColor(_color)
              .setAmbientColor(Color3::fromHsv({_color.hue(), 1.0f, 0.3f}))
              .setTransformationMatrix(_transformation)
              .setNormalMatrix(_transformation.normalMatrix())
              .setProjectionMatrix(projection)
              .draw(_meshs[pos]);
        }
      }
    }
  }

  swapBuffers();
  redraw();
}
void GameOfLife::update()
{
  using namespace std::chrono_literals;
  while (_running) { 
    _playground.nextPlayGround();
    std::this_thread::sleep_for(250ms);
  }
};

void GameOfLife::pointerReleaseEvent(PointerEvent& event)
{
  if (!event.isPrimary() ||
      !(event.pointer() & (Pointer::MouseLeft | Pointer::Finger)))
    return;

  _color = Color3::fromHsv({_color.hue() + 50.0_degf, 1.0f, 1.0f});

  event.setAccepted();
  redraw();
}

void GameOfLife::pointerMoveEvent(PointerMoveEvent& event)
{
  if (!event.isPrimary() ||
      !(event.pointers() & (Pointer::MouseLeft | Pointer::Finger)))
    return;

  Vector2 delta =
      3.0f * Vector2{event.relativePosition()} / Vector2{windowSize()};

  _transformation = Matrix4::rotationX(Rad{delta.y()}) * _transformation *
                    Matrix4::rotationY(Rad{delta.x()});

  event.setAccepted();
  redraw();
}

void GameOfLife::keyPressEvent(KeyEvent& event)
{
  if (event.key() == KeyEvent::Key::Left) {
    Debug{} << "Links gedrückt!";
    _rotate -= 1;
    event.setAccepted(true);
  }
  else if (event.key() == KeyEvent::Key::Right) {
    Debug{} << "Rechts gedrückt!";
    _rotate += 1;
    event.setAccepted(true);
  }
}

void GameOfLife::scrollEvent(ScrollEvent& event)
{
  Vector2 delta = event.offset();  // Scroll-Delta
  float y = delta.y();             // Meist relevant: y

  if (y > 0.0f) {
    _offset += y;
    Debug{} << _offset;
    Debug{} << "Zoom IN";
    // Kamera näher ran
  }
  else if (y < 0.0f) {
    _offset += y;
    Debug{} << _offset;
    Debug{} << "Zoom OUT";
    // Kamera weiter weg
  }

    event.setAccepted();
}  // namespace Project
}  // namespace Project
}  // namespace Magnum

MAGNUM_APPLICATION_MAIN(Magnum::Project::GameOfLife)
