// TODO Integrate: ht//With fast
// Raytracingtps://doc.magnum.graphics/magnum/getting-started.html With fast
// Raytracing 2D/3D
// Time Measurement Multithreaded Configurable
// Compareable
// Visualisieren mit
// https://github.com/bkaradzic/bgfx#include <algorithm>
// https://github.com/mosra/magnum-examples/blob/master/src/primitives/PrimitivesExample.cpp
#include <memory.h>

#include <iostream>
#include <random>
#include <vector>

struct Point2D {
  size_t x;
  size_t y;
};

template <typename T = int>
class Field {
  public:
      Field(const size_t width, const size_t heigth)
          : mwidth(width), mheigth(heigth)
      {
        this->mfield = std::vector<T>(width * heigth);
  }

  Field& operator=(const Field& f)
  {
    *this = Field(f.mwidth, f.mheigth);
    this->mfield = f.mfield;
    return *this;
  }

  auto getValue(const Point2D& p) const
  {
    return this->mfield[p.x + mwidth * p.y];
  }

  void setValue(int val, const Point2D& p)
  {
    this->mfield[p.x + mwidth * p.y] = val;
  }

  auto getNeighbours(const Point2D& p) const
  {
    size_t neighbours = 0;

    if (getValue({p.x - 1, p.y - 1}) != 0) {
      ++neighbours;
    }

    if (getValue({p.x, p.y - 1}) != 0) {
      ++neighbours;
    }

    if (getValue({p.x + 1, p.y - 1}) != 0) {
      ++neighbours;
    }

    if (getValue({p.x - 1, p.y}) != 0) {
      ++neighbours;
    }

    if (getValue({p.x + 1, p.y}) != 0) {
      ++neighbours;
    }

    if (getValue({p.x - 1, p.y + 1}) != 0) {
      ++neighbours;
    }

    if (getValue({p.x, p.y + 1}) != 0) {
      ++neighbours;
    }

    if (getValue({p.x + 1, p.y + 1}) != 0) {
      ++neighbours;
    }

    return neighbours;
  }

  auto getWidth() const { return mwidth; }
  auto getHeigth() const { return mheigth; }
  auto getField() const { return mfield; }

  template <typename C>
  void setField(const std::vector<C>& f)
  {
    this->mfield = f;
  }

  private:
  std::vector<int> mfield;
  const size_t mwidth;
  const size_t mheigth;
};

template <class T = Field<int>>
class PlayGround {
  public:
      PlayGround(size_t width, size_t heigth)
          : mcurrentField(width, heigth), moldField(width, heigth)
      {
        fillPlayGround();
        this->moldField.setField(mcurrentField.getField());
      }

      auto getValue(const Point2D& p) const
      {
        return this->mcurrentField.getValue(p);
      }

      auto getPlayGround() const { return mcurrentField; }

      void nextPlayGround()
      {
        const auto width = this->mcurrentField.getWidth();
        const auto heigth = this->mcurrentField.getHeigth();

        for (size_t x = 1; x < width - 1; ++x) {
          for (size_t y = 1; y < heigth - 1; ++y) {
            const auto neighbours = this->mcurrentField.getNeighbours({x, y});
            if (this->mcurrentField.getValue({x, y}) == 0 && neighbours == 3) {
              this->moldField.setValue(1, {x, y});
            }
            if (this->mcurrentField.getValue({x, y}) == 1 &&
                (neighbours <= 1 || neighbours >= 4)) {
              this->moldField.setValue(0, {x, y});
            }
          }
        }

        swapPlayGrounds();
      }

  private:
      void swapPlayGrounds() { mcurrentField.setField(moldField.getField()); }

      void fillPlayGround()
      {
        const auto width = this->mcurrentField.getWidth();
        const auto heigth = this->mcurrentField.getHeigth();

        std::mt19937 mt{};

        for (size_t x = 0; x < width; ++x) {
          for (size_t y = 0; y < heigth; ++y) {
            auto value = std::rand() % 2;
            this->mcurrentField.setValue(value, {x, y});
          }
        }
      }

      T mcurrentField;
      T moldField;
};

struct Point3D {
  size_t x;
  size_t y;
  size_t z;
};

template <typename T = int>
class Field3D {
  public:
      Field3D(const size_t width, const size_t heigth, const size_t deep)
          : mwidth(width), mheigth(heigth), mdeep(deep)
      {
        this->mfield = std::vector<T>(width * heigth * deep);
  }

  Field3D& operator=(const Field3D& f)
  {
    *this = Field3D(f.mwidth, f.mheigth, f.deep);
    this->mfield = f.mfield;
    return *this;
  }

  auto getValue(const Point3D& p) const
  {
    return this->mfield[p.x + mwidth * p.y + (mheigth + mwidth) * p.z];
  }

  void setValue(int val, const Point3D& p)
  {
    this->mfield[p.x + mwidth * p.y + (mheigth + mwidth) * p.z] = val;
  }

  auto getNeighbours(const Point3D& p, const size_t offsetz) const
  {
    size_t neighbours = 0;

    if (getValue({p.x - 1, p.y - 1, p.z + offsetz}) != 0) {
      ++neighbours;
    }

    if (getValue({p.x, p.y - 1, p.z + offsetz}) != 0) {
      ++neighbours;
    }

    if (getValue({p.x + 1, p.y - 1, p.z + offsetz}) != 0) {
      ++neighbours;
    }

    if (getValue({p.x - 1, p.y, p.z + offsetz}) != 0) {
      ++neighbours;
    }

    if (getValue({p.x + 1, p.y, p.z + offsetz}) != 0) {
      ++neighbours;
    }

    if (offsetz != 0) {
      if (getValue({p.x, p.y, p.z + offsetz}) != 0) {
        ++neighbours;
      }
    }

    if (getValue({p.x - 1, p.y + 1, p.z + offsetz}) != 0) {
      ++neighbours;
    }

    if (getValue({p.x, p.y + 1, p.z + offsetz}) != 0) {
      ++neighbours;
    }

    if (getValue({p.x + 1, p.y + 1, p.z + offsetz}) != 0) {
      ++neighbours;
    }

    return neighbours;
  }

  auto getWidth() const { return mwidth; }
  auto getHeigth() const { return mheigth; }
  auto getDeep() const { return mdeep; }
  auto getField() const { return mfield; }

  template <typename C>
  void setField(const std::vector<C>& f)
  {
    this->mfield = f;
  }

  private:
      std::vector<int> mfield;
      const size_t mwidth;
      const size_t mheigth;
      const size_t mdeep;
};

template <class T = Field3D<int>>
class PlayGround3D {
  public:
      PlayGround3D(size_t width, size_t heigth, size_t deep)
          : mcurrentField(width, heigth, deep), moldField(width, heigth, deep)
      {
        fillPlayGround();
        this->moldField.setField(mcurrentField.getField());
      }

      auto getValue(const Point3D& p) const
      {
        return this->mcurrentField.getValue(p);
      }

      auto getPlayGround() const { return this->mcurrentField; }

      void nextPlayGround()
      {
        const auto width = this->mcurrentField.getWidth();
        const auto heigth = this->mcurrentField.getHeigth();
        const auto deep = this->mcurrentField.getDeep();

        for (size_t z = 1; z < deep - 1; ++z) {
          for (size_t x = 1; x < width - 1; ++x) {
            for (size_t y = 1; y < heigth - 1; ++y) {
              size_t neighbours = 0;
              neighbours += this->mcurrentField.getNeighbours({x, y, z}, -1);
              neighbours += this->mcurrentField.getNeighbours({x, y, z}, 0);
              neighbours += this->mcurrentField.getNeighbours({x, y, z}, 1);
              // check rule
              if (this->mcurrentField.getValue({x, y, z}) == 1 &&
                  (neighbours < 13 || neighbours > 19)) {
                this->moldField.setValue(0, {x, y, z});
              }
              if (this->mcurrentField.getValue({x, y, z}) == 0 &&
                  (neighbours >= 14 || neighbours <= 19)) {
                this->moldField.setValue(1, {x, y, z});
              }
           }
          }
        }
        swapPlayGrounds();
      }

  private:
      void swapPlayGrounds() { mcurrentField.setField(moldField.getField()); }

      void fillPlayGround()
      {
        std::mt19937 mt{};

        const auto width = this->mcurrentField.getWidth();
        const auto heigth = this->mcurrentField.getHeigth();
        const auto deep = this->mcurrentField.getDeep();

        for (size_t z = 0; z < width; ++z) {
          for (size_t x = 0; x < width; ++x) {
            for (size_t y = 0; y < heigth; ++y) {
              auto value = std::rand() % 2;
              this->mcurrentField.setValue(value, {x, y, z});
            }
          }
        }
      }

      T mcurrentField;
      T moldField;
};

void showPlayGround(const PlayGround<Field<int>>& mground)
{
  const auto width = mground.getPlayGround().getWidth();
  const auto heigth = mground.getPlayGround().getHeigth();

  for (size_t x = 0; x < width; ++x) {
    for (size_t y = 0; y < heigth; ++y) {
      std::cout << mground.getValue({x, y});
    }
    std::cout << "\n";
  }
}

void showPlayGround(const PlayGround3D<Field3D<int>>& mground)
{
  const auto width = mground.getPlayGround().getWidth();
  const auto heigth = mground.getPlayGround().getHeigth();
  const auto deep = mground.getPlayGround().getDeep();

  for (size_t x = 0; x < width; ++x) {
    for (size_t y = 0; y < heigth; ++y) {
      std::cout << mground.getValue({x, y, 0});
    }
    std::cout << "\n";
  }
}

struct PlayDefinition2D {
  size_t widht;
  size_t heigth;
};

struct PlayDefinition3D {
  size_t widht;
  size_t heigth;
  size_t deep;
};

auto makePlayGround(const PlayDefinition2D& pd)
{
  return PlayGround(pd.widht, pd.heigth);
}

auto makePlayGround(const PlayDefinition3D& pd)
{
  return PlayGround3D(pd.widht, pd.heigth, pd.deep);
}

int main(int argc, char** argv)
{
  std::string cmd = "";
  int iterations = 10;

  if (argc > 1) {
    cmd = std::string(argv[1]);
  }

  if (argc > 2) {
    iterations = std::atoi(argv[2]);
  }

 if (cmd == "2D") {
    auto mPlayGround = makePlayGround(PlayDefinition2D{100, 100});
    for (int i = 0; i < iterations; ++i) {
      mPlayGround.nextPlayGround();
      std::cout << "New Iteration" << i << std::endl;
      showPlayGround(mPlayGround);
      std::cout << std::endl;
    }
  }
  else if (cmd == "3D") {
    auto mPlayGround = makePlayGround(PlayDefinition3D{100, 100, 100});
    for (int i = 0; i < iterations; ++i) {
      mPlayGround.nextPlayGround();
      std::cout << "New Iteration" << i << std::endl;
      showPlayGround(mPlayGround);
      std::cout << std::endl;
    }
  }

     return 0;
}
