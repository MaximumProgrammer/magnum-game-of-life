/*Markus Gruber
 *Markus.Gruber4@gmx.net
 */

#include <memory.h>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "Queue_Safe.h"

class ThreadPool {
  public:
  explicit ThreadPool(int numThreads) : m_threads(numThreads), stop(false)
  {
    for (int i = 0; i < m_threads; i++) {
      threads.emplace_back([this] {
        std::function<void()> task;
        while (1) {
          std::unique_lock<std::mutex> lock(mtx);
          cv.wait(lock, [this] { return !tasks.empty() || stop; });
          if (stop) return;
          task = std::move(tasks.pop());
          lock.unlock();
          task();
        }
      });
    }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool(){
      std::unique_lock<std::mutex> lock(mtx);
      stop = true;
      lock.unlock();
      cv.notify_all();

      for(auto& th: threads)
      {
        th.join();
      }
    }

    template <class F, class... Args>
    auto ExecuteTask(F&& f, Args&&... args) -> std::future<decltype(f(args...))>
    {
      using return_type = decltype(f(args...));

      auto task = std::make_shared<std::packaged_task<return_type()>>(
          std::bind(std::forward<F>(f), std::forward<Args>(args)...));

      std::future<return_type> res = task->get_future();

      std::unique_lock<std::mutex> lock(mtx);
      tasks.push([task]() -> void { (*task)(); });
      lock.unlock();
      cv.notify_one();
      return res;
    }

    auto running_tasks() { return tasks.size(); }

private:
    int m_threads;
    std::vector<std::thread> threads;
    Queue_Safe<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop;
};

struct PlayDefinition2D {
  size_t widht;
  size_t heigth;
};

struct PlayDefinition3D {
  size_t widht;
  size_t heigth;
  size_t deep;
};

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

  Field(const Field& f) : mwidth(f.mwidth), mheigth(f.mheigth)
  {
    this->mfield = f.mfield;
  }

  Field& operator=(const Field& f)
  {
    *this = Field(f.mwidth, f.mheigth);
    this->mfield = f.mfield;
    return *this;
  }

  auto getValue(const Point2D& p) const
  {
    std::lock_guard<std::mutex> guard(mtx);
    return this->mfield[p.x + mwidth * p.y];
  }

  void setValue(int val, const Point2D& p)
  {
    std::lock_guard<std::mutex> guard(mtx);
    this->mfield[p.x + mwidth * p.y] = val;
  }

  auto getNeighbours(const Point2D& p) const
  {
    size_t neighbours = 0;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        if (dx == 0 && dy == 0) continue;
        neighbours += getValue({p.x + dx, p.y + dy});
      }
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
  mutable std::mutex mtx;
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

  Field3D(const Field3D& f)
      : mwidth(f.mwidth), mheigth(f.mheigth), mdeep(f.mdeep)
  {
    this->mfield = f.mfield;
  }

  Field3D& operator=(const Field3D& f)
  {
    *this = Field3D(f.mwidth, f.mheigth, f.mdeep);
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

  auto getNeighbours(const Point3D& p) const
  {
    size_t neighbours = 0;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0) continue;
          neighbours += getValue({p.x + dx, p.y + dy, p.z + dz});
        }
      }
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
  mutable std::mutex mtx;
};

template <class T = Field3D<int>>
class PlayGround3D {
  public:
      PlayGround3D(size_t width, size_t heigth, size_t deep)
          : mcurrentField(width, heigth, deep),
            moldField(width, heigth, deep),
            mpool(16)
      {
        fillPlayGround();
        this->moldField.setField(mcurrentField.getField());
      }

       size_t size()
       {
        const auto width = this->mcurrentField.getWidth();
        const auto heigth = this->mcurrentField.getHeigth();
        const auto deep = this->mcurrentField.getDeep();
        return width * heigth * deep;
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

            /*SERIAL*/
            /*
            for (size_t z = 1; z < deep - 1; ++z) {
              for (size_t x = 1; x < width - 1; ++x) {
                for (size_t y = 1; y < heigth - 1; ++y) {
                  size_t neighbours = 0;
                  neighbours += this->mcurrentField.getNeighbours({x, y, z},
            -1); neighbours += this->mcurrentField.getNeighbours({x, y, z}, 0);
                  neighbours += this->mcurrentField.getNeighbours({x, y, z}, 1);
                  // check rule
                  if (this->mcurrentField.getValue({x, y, z}) == 1 &&
                      (neighbours <= 13 || neighbours > 19)) {
                    this->moldField.setValue(0, {x, y, z});
                  }
                  if (this->mcurrentField.getValue({x, y, z}) == 0 &&
                      (neighbours >= 14 || neighbours <= 19)) {
                    this->moldField.setValue(1, {x, y, z});
                  }
                }
              }
            }
             */
            /*PARALLEL*/
            // https://www.geeksforgeeks.org/cpp/thread-pool-in-cpp/
            // https://medium.com/@bhushanrane1992/getting-started-with-c-thread-pool-b6d1102da99a
            std::cout << "New Iteration" << std::endl;
            for (size_t z = 1; z < deep - 1; ++z) {
              this->mpool.ExecuteTask([indexz = z, this, width, heigth]() {
                for (size_t x = 1; x < width - 1; ++x) {
                  for (size_t y = 1; y < heigth - 1; ++y) {
                    size_t neighbours = this->mcurrentField.getNeighbours(
                        Point3D{x, y, indexz});
                    if (this->mcurrentField.getValue({x, y, indexz}) == 1 &&
                        (neighbours <= 13 || neighbours > 19)) {
                      this->moldField.setValue(0, {x, y, indexz});
                    }
                    if (this->mcurrentField.getValue({x, y, indexz}) == 0 &&
                        (neighbours >= 14 || neighbours <= 19)) {
                      this->moldField.setValue(1, {x, y, indexz});
                    }
                  }
                }
              });
            }
            while (mpool.running_tasks() != 0) {
            }
            std::cout << "Swap" << std::endl;
            swapPlayGrounds();
          }

      private:
          void swapPlayGrounds()
          {
            mcurrentField.setField(moldField.getField());
          }

          void fillPlayGround()
          {
            std::mt19937 mt{};

            const auto width = this->mcurrentField.getWidth();
            const auto heigth = this->mcurrentField.getHeigth();
            const auto deep = this->mcurrentField.getDeep();

            for (size_t z = 0; z < deep; ++z) {
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
          ThreadPool mpool;
        };

        auto makePlayGround(const PlayDefinition2D& pd)
        {
          return PlayGround(pd.widht, pd.heigth);
        }

        auto makePlayGround(const PlayDefinition3D& pd)
        {
          return PlayGround3D(pd.widht, pd.heigth, pd.deep);
        }

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
              std::cout << mground.getValue({x, y, deep / 2 + 1});
            }
            std::cout << "\n";
          }
        }


