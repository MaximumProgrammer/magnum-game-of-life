/* Markus Gruber
 * Markus.Gruber4@gmx.net
 */
#include <memory.h>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "GameOfLifeCUDA3d.h"
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

template <typename T = size_t>
struct Point2D {
  T x;
  T y;
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

  auto getValue(const Point2D<>& p) const
  {
    std::lock_guard<std::mutex> guard(mtx);
    return this->mfield[p.x + mwidth * p.y];
  }

  void setValue(int val, const Point2D<>& p)
  {
    std::lock_guard<std::mutex> guard(mtx);
    this->mfield[p.x + mwidth * p.y] = val;
  }

  auto getNeighbours(const Point2D<>& p) const
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

      auto getValue(const Point2D<>& p) const
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
template <typename T = size_t>
struct Point3D {
  T x;
  T y;
  T z;
};

template <typename T = int>
class Field3D {
  public:
      Field3D(const size_t width, const size_t heigth, const size_t deep)
          : mwidth(width), mheigth(heigth), mdeep(deep)
      {
        this->mfield = std::vector<T>(width * heigth * deep);
  }

  Field3D(const Field3D<>& f)
      : mwidth(f.mwidth), mheigth(f.mheigth), mdeep(f.mdeep)
  {
    this->mfield = f.mfield;
  }

  Field3D& operator=(const Field3D<>& f)
  {
    *this = Field3D(f.mwidth, f.mheigth, f.mdeep);
    this->mfield = f.mfield;
    return *this;
  }

  auto getValue(const Point3D<>& p) const
  {
    return this->mfield[p.x + mwidth * p.y + (mheigth + mwidth) * p.z];
  }

  void setValue(int val, const Point3D<>& p)
  {
    this->mfield[p.x + mwidth * p.y + (mheigth + mwidth) * p.z] = val;
  }

  auto getNeighbours(const Point3D<>& p) const
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

  auto& getValues() { return mfield; }

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
      enum ComputeMode {
        ComputeCPUSerial,
        ComputeCPUParallel,
        ComputeGPUParallel
      };
      PlayGround3D(const size_t width, const size_t heigth, const size_t deep,
                   const ComputeMode mode = ComputeMode::ComputeCPUParallel)
          : mcurrentField(width, heigth, deep),
            moldField(width, heigth, deep),
            mmode(mode),
            mpool(16),
            mgpuavailable(gpu_available())
      {
        fillPlayGround();
        this->moldField.setField(mcurrentField.getField());
        if (mgpuavailable) {
          cuda_init(width, heigth, deep);
        }
      }
      ~PlayGround3D()
      {
        if (mgpuavailable) {
          cuda_free();
        }
      }
      void setComputingMode(ComputeMode mode) { mmode = mode; }

      size_t size()
      {
        const auto width = this->mcurrentField.getWidth();
        const auto heigth = this->mcurrentField.getHeigth();
        const auto deep = this->mcurrentField.getDeep();
        return width * heigth * deep;
       }

       auto getValue(const Point3D<>& p) const
       {
         return this->mcurrentField.getValue(p);
       }

       auto getPlayGround() const { return this->mcurrentField; }

           void nextPlayGround()
       {
         using clock = std::chrono::system_clock;
         using sec = std::chrono::duration<double>;
         const auto before = clock::now();

         /*SERIAL*/
         std::string mode = "";
         if (mmode == PlayGround3D::ComputeMode::ComputeCPUSerial) {
           computeCPUSerial();
           mode = "CPUSerial";
         }
         /*PARALLEL*/
         if (mmode == PlayGround3D::ComputeMode::ComputeCPUParallel) {
           computeCPUParallel();
           mode = "CPUParallel";
         }
         /*CUDA*/
         if (mmode == PlayGround3D::ComputeMode::ComputeGPUParallel) {
           computeGPUParallel();
           mode = "GPUParallel";
         }
         const sec duration = clock::now() - before;
         std::cout << "Computing took " << duration.count() << "s"
                   << " with mode: " << mode << std::endl;
       }

   private:
       void swapPlayGrounds() { mcurrentField.setField(moldField.getField()); }
       void computeCPUSerial()
       {
         const auto width = this->mcurrentField.getWidth();
         const auto heigth = this->mcurrentField.getHeigth();
         const auto deep = this->mcurrentField.getDeep();

          for (size_t z = 1; z < deep - 1; ++z) {
           for (size_t x = 1; x < width - 1; ++x) {
             for (size_t y = 1; y < heigth - 1; ++y) {
               size_t neighbours =
                   this->mcurrentField.getNeighbours(Point3D{x, y, z});
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
         swapPlayGrounds();
       };

       void computeCPUParallel()
       {
         const auto width = this->mcurrentField.getWidth();
         const auto heigth = this->mcurrentField.getHeigth();
         const auto deep = this->mcurrentField.getDeep();

         for (size_t z = 1; z < deep - 1; ++z) {
           this->mpool.ExecuteTask([indexz = z, this, width, heigth]() {
             for (size_t x = 1; x < width - 1; ++x) {
               for (size_t y = 1; y < heigth - 1; ++y) {
                 size_t neighbours =
                     this->mcurrentField.getNeighbours(Point3D{x, y, indexz});
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
         swapPlayGrounds();
       };
       void computeGPUParallel()
       {
         if (mgpuavailable) {
           const auto width = this->mcurrentField.getWidth();
           const auto heigth = this->mcurrentField.getHeigth();
           const auto deep = this->mcurrentField.getDeep();

           next_step(this->moldField.getValues(),
                     this->mcurrentField.getValues(), width, heigth, deep);
         }
         else {
           std::cout << "No GPU found falling back to PARALLEL Mode"
                     << std::endl;
           computeCPUParallel();
         }
         swapPlayGrounds();
       };

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
       ComputeMode mmode;
       ThreadPool mpool;
       const bool mgpuavailable;
};

auto inline makePlayGround(const PlayDefinition2D& pd)
        {
          return PlayGround(pd.widht, pd.heigth);
        }

        auto inline makePlayGround(const PlayDefinition3D& pd)
        {
          return PlayGround3D(pd.widht, pd.heigth, pd.deep);
        }

        void inline showPlayGround(const PlayGround<Field<int>>& mground)
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

        void inline showPlayGround(const PlayGround3D<Field3D<int>>& mground)
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


