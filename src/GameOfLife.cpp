/* Markus Gruber
 * Markus.Gruber4@gmx.net
 */

#include <Corrade/Containers/Optional.h>
#include <Corrade/Utility/Arguments.h>
#include <Magnum/DebugTools/ColorMap.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/ImageView.h>
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Time.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/Shaders/MeshVisualizerGL.h>
#include <Magnum/Shaders/PhongGL.h>
#include <Magnum/Trade/MeshData.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "ArcBall.h"
#include "ArcBallCamera.h"
#include "BackendGameOfLife.h"

namespace Magnum { namespace Examples {

using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
using Scene3D = SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;

using namespace Math::Literals;

class GameOfLife3D : public Platform::Application {
  public:
      explicit GameOfLife3D(const Arguments& arguments);

  private:
    void drawEvent() override;
    void viewportEvent(ViewportEvent& event) override;
    void keyPressEvent(KeyEvent& event) override;
    void pointerPressEvent(PointerEvent& event) override;
    void pointerReleaseEvent(PointerEvent& event) override;
    void pointerMoveEvent(PointerMoveEvent& event) override;
    void scrollEvent(ScrollEvent& event) override;
    void update();
    PlayGround3D<Field3D<int>> createPlayGroundFromArguments(const Arguments& arguments);

    Scene3D _scene;
    SceneGraph::DrawableGroup3D _drawables;
    PlayGround3D<Field3D<int>> _playground;

    std::vector<GL::Mesh> _meshs;
    Shaders::PhongGL _shader{NoCreate};
    Containers::Optional<ArcBallCamera> _arcballCamera;
    bool _running;
    std::thread _thread;
};

template <class S, class M, class P>
class VisualizationDrawable : public SceneGraph::Drawable3D {
  public:
      explicit VisualizationDrawable(Object3D& object, S& shader, M& meshs,
                                     P& playground,
                                     SceneGraph::DrawableGroup3D& drawables)
          : SceneGraph::Drawable3D{object, &drawables},
            _shader(shader),
            _meshs(meshs),
            _playground(playground)
      {
      }

  void draw(const Matrix4& transformation, SceneGraph::Camera3D& camera)
  {
    const auto width = _playground.getPlayGround().getWidth();
    const auto heigth = _playground.getPlayGround().getHeigth();
    const auto deep = _playground.getPlayGround().getDeep();

    const float step = 5.0f; 
    const auto center = Point3D<float>(-static_cast<float>(width * step / 2),
                                       -static_cast<float>(heigth * step / 2),
                                       -static_cast<float>(deep * step / 2));
    for (size_t i = 0; i < width; ++i) {
      for (size_t j = 0; j < heigth; ++j) {
        for (size_t k = 0; k < deep; ++k) {
          if (const auto isAlive = _playground.getValue({i, j, k}); isAlive) {
            auto _transformation =
                transformation +
                Matrix4::translation({center.x + step * i, center.y + step * j,
                                      center.z + step * k});

            const auto pos = i + width * j + (width + heigth) * k;
            _shader.draw(_meshs[pos])
                .setTransformationMatrix(_transformation)
                .setProjectionMatrix(camera.projectionMatrix())
                .setNormalMatrix(_transformation.normalMatrix());
          }
        }
      }
    }
  }

  private:
      S& _shader;
      M& _meshs;
      P& _playground;
};

GameOfLife3D::GameOfLife3D(const Arguments& arguments)
    : Platform::Application{arguments, Configuration{}
                                           .setTitle("Magnum Game Of Life 3D")
                                           .setSize({1200, 1000})},
      _playground(createPlayGroundFromArguments(arguments)),
      _running(false)
{
  GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
  GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

  {
    for (size_t i = 0; i < _playground.size(); ++i) {
      auto mesh = MeshTools::compile(Primitives::cubeSolid());
      _meshs.push_back(std::move(mesh));
    }

    _shader =
        Shaders::PhongGL{Shaders::PhongGL::Configuration{}.setLightCount(1)};

    auto _color = Color3::fromHsv({35.0_degf, 1.0f, 1.0f});

    _shader.setLightPositions({{1.4f, 1.0f, 0.75f, 0.0f}})
        .setDiffuseColor(_color)
        .setAmbientColor(Color3::fromHsv({_color.hue(), 1.0f, 0.3f}));

    auto object = new Object3D{&_scene};
    (*object).rotateY(40.0_degf).rotateX(-30.0_degf);

    new VisualizationDrawable<decltype(_shader), decltype(_meshs),
                              decltype(_playground)>{*object, _shader, _meshs,
                                                     _playground, _drawables};
  }

  /* Set up the camera */
  {
    /* Setup the arcball after the camera objects */
    const Vector3 eye = Vector3::zAxis(-200.0f);
    const Vector3 center{};
    const Vector3 up = Vector3::yAxis();
    _arcballCamera.emplace(_scene, eye, center, up, 45.0_degf, windowSize(),
                           framebufferSize());
  }

  /* Loop at 60 Hz max */
  setSwapInterval(1);
  setMinimalLoopPeriod(16.0_msec);
  _running = true;
  _thread = std::thread(&GameOfLife3D::update, this);
}

PlayGround3D<Field3D<int>> GameOfLife3D::createPlayGroundFromArguments(
    const Arguments& arguments)
{
  Utility::Arguments args;
  args.addOption("dimension", "16")
      .setHelp("dimension", "number of blocks in each direction", "blocks")
      .addOption("computemode", "2")
      .setHelp("computemode", "how the next step should be computed", "mode")
      .addSkippedPrefix("magnum")
      .parse(arguments.argc, arguments.argv);

  if (args.value<UnsignedInt>("computemode") == 1) {
    auto dimension = args.value<UnsignedShort>("dimension");
    return PlayGround3D<>(dimension, dimension, dimension,
                          PlayGround3D<>::ComputeMode::ComputeCPUSerial);
  }
  else if (args.value<UnsignedInt>("computemode") == 2) {
    auto dimension = args.value<UnsignedShort>("dimension");
    return PlayGround3D<>(dimension, dimension, dimension,
                          PlayGround3D<>::ComputeMode::ComputeCPUParallel);
  }
  else if (args.value<UnsignedInt>("computemode") == 3) {
    auto dimension = args.value<UnsignedShort>("dimension");
    return PlayGround3D<>(dimension, dimension, dimension,
                          PlayGround3D<>::ComputeMode::ComputeGPUParallel);
  }
  return PlayGround3D<>(32, 32, 32,
                        PlayGround3D<>::ComputeMode::ComputeCPUSerial);
}
void GameOfLife3D::drawEvent()
{
  GL::defaultFramebuffer.clear(GL::FramebufferClear::Color |
                               GL::FramebufferClear::Depth);

  /* Call arcball update in every frame. This will do nothing if the
     camera has not been changed. Otherwise, camera transformation
     will be propagated into the camera objects. */
  [[maybe_unused]] bool camChanged = _arcballCamera->update();
  _arcballCamera->draw(_drawables);
  swapBuffers();

  /*if (camChanged)*/ redraw();
}

void GameOfLife3D::viewportEvent(ViewportEvent& event)
{
  GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});

  _arcballCamera->reshape(event.windowSize(), event.framebufferSize());

  redraw(); /* camera has changed, redraw! */
}

void GameOfLife3D::keyPressEvent(KeyEvent& event)
{
  switch (event.key()) {
    case Key::L:
      if (_arcballCamera->lagging() > 0.0f) {
        Debug{} << "Lagging disabled";
        _arcballCamera->setLagging(0.0f);
      }
      else {
        Debug{} << "Lagging enabled";
        _arcballCamera->setLagging(0.85f);
      }
      break;
    case Key::R:
      _arcballCamera->reset();
      break;

    default:
      return;
  }

  event.setAccepted();
  redraw(); /* camera has changed, redraw! */
}

void GameOfLife3D::pointerPressEvent(PointerEvent& event)
{
  if (!event.isPrimary() ||
      !(event.pointer() & (Pointer::MouseLeft | Pointer::Finger)))
    return;

  /* Enable mouse capture so the mouse can drag outside of the window
   */
  /** @todo replace once https://github.com/mosra/magnum/pull/419 is
   * in */
  SDL_CaptureMouse(SDL_TRUE);

  _arcballCamera->initTransformation(event.position());

  event.setAccepted();
  redraw(); /* camera has changed, redraw! */
}

void GameOfLife3D::pointerReleaseEvent(PointerEvent& event)
{
  if (!event.isPrimary() ||
      !(event.pointer() & (Pointer::MouseLeft | Pointer::Finger)))
    return;

  /* Disable mouse capture again */
  /** @todo replace once https://github.com/mosra/magnum/pull/419 is
   * in */
  SDL_CaptureMouse(SDL_FALSE);

  event.setAccepted();
  redraw();
}

void GameOfLife3D::pointerMoveEvent(PointerMoveEvent& event)
{
  if (!event.isPrimary() ||
      !(event.pointers() & (Pointer::MouseLeft | Pointer::Finger)))
    return;

  if (event.modifiers() & Modifier::Shift)
    _arcballCamera->translate(event.position());
  else
    _arcballCamera->rotate(event.position());

  event.setAccepted();
  redraw(); /* camera has changed, redraw! */
}

void GameOfLife3D::scrollEvent(ScrollEvent& event)
{
  const Float delta = event.offset().y();
  if (Math::abs(delta) < 1.0e-2f) return;

  _arcballCamera->zoom(delta);

  event.setAccepted();
  redraw(); /* camera has changed, redraw! */
}

void GameOfLife3D::update()
{
  using namespace std::chrono_literals;

  while (_running) {
    _playground.nextPlayGround();
    std::this_thread::sleep_for(1000ms);
    redraw();
  }
};

}  // namespace Examples
}  // namespace Magnum

MAGNUM_APPLICATION_MAIN(Magnum::Examples::GameOfLife3D)
