/*Markus Gruber
 *Markus.Gruber4@gmx.net
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

bool gpu_available()
{
  int deviceCount = 0;
#ifdef __CUDACC__
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  if (err != cudaSuccess || deviceCount == 0) {
    std::cout << "Keine CUDA-faehige GPU gefunden\n";
    return false;
  }
  else {
    std::cout << deviceCount << " CUDA-GPU(s) gefunden\n";
    return true;
  }
#endif
  return deviceCount;
}

#ifdef __CUDACC__
#include <cuda_runtime.h>

static int* d_current = nullptr;
static int* d_next = nullptr;

__device__ __host__ inline int idx(int x, int y, int WIDTH)
{
  return x + WIDTH * y;
}

__global__ void lifeStep2D(const int* current, int* next, int WIDTH, int HEIGHT)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= WIDTH || y >= HEIGHT) return;

  int neighbors = 0;

  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0)
            continue;

        int nx = x + dx;
        int ny = y + dy;

        if (nx >= 0 && nx < WIDTH &&
            ny >= 0 && ny < HEIGHT) {
          neighbors += current[idx(nx, ny, WIDTH)];
        }
    }

  uint8_t alive = current[idx(x, y, WIDTH)];

  // Conway: B3 / S23
  if (!alive && neighbors == 3)
    next[idx(x, y, WIDTH)] = 1;
  else if (alive && (neighbors == 2 || neighbors == 3))
    next[idx(x, y, WIDTH)] = 1;
  else
    next[idx(x, y, WIDTH)] = 0;
}

void cuda_init(const size_t NX, const size_t NY)
{
  const int SIZE = NX * NY;
  cudaMalloc(&d_current, SIZE * sizeof(int));
  cudaMalloc(&d_next, SIZE * sizeof(int));
}

void cuda_free()
{
  cudaFree(d_current);
  cudaFree(d_next);
}

void next_step(std::vector<int>& h_next, const std::vector<int>& h_current,
               const size_t NX, const size_t NY)
{
  const auto SIZE = NX * NY;
  cudaMalloc(&d_current, SIZE * sizeof(int));
  cudaMalloc(&d_next, SIZE * sizeof(int));

  cudaMemcpy(d_current, h_current.data(), SIZE * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) printf("Error 1: %s\n", cudaGetErrorString(err));

  dim3 block(16, 16);
  dim3 grid((NX + block.x - 1) / block.x, (NY + block.y - 1) / block.y);

  lifeStep2D<<<grid, block, 48>>>(d_current, d_next, NX, NY);
  if (err != cudaSuccess) printf("Error 2: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();

  cudaMemcpy(h_next.data(), d_next, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) printf("Error 2: %s\n", cudaGetErrorString(err));
}
#endif
