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

__device__ __host__ inline int idx(int x, int y, int z, int NX, int NY, int NZ)
{
  return x + NX * (y + NY * z);
}

__global__ void lifeStep(int* current, int* next, int NX, int NY, int NZ)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= NX || y >= NY || z >= NZ) return;

  int neighbors = 0;

  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0) continue;

        int nx = x + dx;
        int ny = y + dy;
        int nz = z + dz;

        if (nx >= 0 && nx < NX && ny >= 0 && ny < NY && nz >= 0 && nz < NZ) {
          neighbors += current[idx(nx, ny, nz, NX, NY, NZ)];
        }
      }

  bool alive = current[idx(x, y, z, NX, NY, NZ)];

  if (alive && (neighbors <= 13 || neighbors > 19))
    next[idx(x, y, z, NX, NY, NZ)] = false;
  else if (!alive && (neighbors >= 14 || neighbors <= 19))
    next[idx(x, y, z, NX, NY, NZ)] = true; 
}

static int* d_current = nullptr;
static int* d_next = nullptr;

void cuda_init(const size_t NX, const size_t NY, const size_t NZ)
{
  const int SIZE = NX * NY * NZ;
  cudaMalloc(&d_current, SIZE * sizeof(int));
  cudaMalloc(&d_next, SIZE * sizeof(int));
}

void cuda_free()
{
  cudaFree(d_current);
  cudaFree(d_next);
}

void next_step(std::vector<int>& h_next, const std::vector<int>& h_current,
               const size_t NX, const size_t NY, const size_t NZ)
{
  const auto SIZE = NX * NY * NZ;
  cudaMemcpy(d_current, h_current.data(), SIZE * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) printf("Error 1: %s\n", cudaGetErrorString(err));

  dim3 block(8, 8, 8);
  dim3 grid((NX + block.x - 1) / block.x, (NY + block.y - 1) / block.y,
            (NZ + block.z - 1) / block.z);

  lifeStep<<<block, block, 48>>>(d_current, d_next, NX, NY, NZ);
  err = cudaGetLastError();
  if (err != cudaSuccess) printf("Error 2: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();

  cudaMemcpy(h_next.data(), d_next, SIZE * sizeof(bool),
             cudaMemcpyDeviceToHost);
}
#endif
