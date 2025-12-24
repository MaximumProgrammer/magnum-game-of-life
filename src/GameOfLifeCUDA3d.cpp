#include <cuda_runtime.h>
#include <iostream>
#include <vector>

constexpr int NX = 32;
constexpr int NY = 32;
constexpr int NZ = 32;

constexpr int SIZE = NX * NY * NZ;

__device__ __host__
inline int idx(int x, int y, int z) {
    return x + NX * (y + NY * z);
}

__global__
void lifeStep(const bool* current, bool* next) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
        return;

    int neighbors = 0;

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0)
            continue;

        int nx = x + dx;
        int ny = y + dy;
        int nz = z + dz;

        if (nx >= 0 && nx < NX &&
            ny >= 0 && ny < NY &&
            nz >= 0 && nz < NZ) {
            neighbors += current[idx(nx, ny, nz)];
        }
    }

    bool alive = current[idx(x, y, z)];

    // B6 / S567
    if (!alive && neighbors == 6)
        next[idx(x, y, z)] = true;
    else if (alive && (neighbors == 5 || neighbors == 6 || neighbors == 7))
        next[idx(x, y, z)] = true;
    else
        next[idx(x, y, z)] = false;
}

int main() {
    std::vector<bool> h_current(SIZE);
    std::vector<bool> h_next(SIZE);

    // Zufällige Initialisierung
    for (int i = 0; i < SIZE; ++i)
        h_current[i] = (rand() % 100) < 25;

    bool *d_current, *d_next;
    cudaMalloc(&d_current, SIZE * sizeof(bool));
    cudaMalloc(&d_next, SIZE * sizeof(bool));

    cudaMemcpy(d_current, h_current.data(),
               SIZE * sizeof(bool), cudaMemcpyHostToDevice);

    dim3 block(8, 8, 8);
    dim3 grid(
        (NX + block.x - 1) / block.x,
        (NY + block.y - 1) / block.y,
        (NZ + block.z - 1) / block.z
    );

    for (int gen = 0; gen < 100; ++gen) {
        lifeStep<<<grid, block>>>(d_current, d_next);
        cudaDeviceSynchronize();

        std::swap(d_current, d_next);
    }

    cudaMemcpy(h_current.data(), d_current,
               SIZE * sizeof(bool), cudaMemcpyDeviceToHost);

    // Debug: lebende Zellen zählen
    int alive = 0;
    for (bool c : h_current) alive += c;
    std::cout << "Alive cells: " << alive << std::endl;

    cudaFree(d_current);
    cudaFree(d_next);
}
