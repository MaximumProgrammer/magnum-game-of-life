#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

constexpr int WIDTH  = 1024;
constexpr int HEIGHT = 1024;
constexpr int SIZE   = WIDTH * HEIGHT;

__device__ __host__
inline int idx(int x, int y) {
    return x + WIDTH * y;
}

__global__
void lifeStep2D(const uint8_t* current, uint8_t* next) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    int neighbors = 0;

    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0)
            continue;

        int nx = x + dx;
        int ny = y + dy;

        if (nx >= 0 && nx < WIDTH &&
            ny >= 0 && ny < HEIGHT) {
            neighbors += current[idx(nx, ny)];
        }
    }

    uint8_t alive = current[idx(x, y)];

    // Conway: B3 / S23
    if (!alive && neighbors == 3)
        next[idx(x, y)] = 1;
    else if (alive && (neighbors == 2 || neighbors == 3))
        next[idx(x, y)] = 1;
    else
        next[idx(x, y)] = 0;
}

int main() {
    std::vector<uint8_t> h_current(SIZE);
    std::vector<uint8_t> h_next(SIZE);

    // Zufällige Initialisierung
    for (int i = 0; i < SIZE; ++i)
        h_current[i] = (rand() % 100) < 20;

    uint8_t *d_current, *d_next;
    cudaMalloc(&d_current, SIZE * sizeof(uint8_t));
    cudaMalloc(&d_next, SIZE * sizeof(uint8_t));

    cudaMemcpy(d_current, h_current.data(),
               SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(
        (WIDTH  + block.x - 1) / block.x,
        (HEIGHT + block.y - 1) / block.y
    );

    for (int gen = 0; gen < 500; ++gen) {
        lifeStep2D<<<grid, block>>>(d_current, d_next);
        cudaDeviceSynchronize();
        std::swap(d_current, d_next);
    }

    cudaMemcpy(h_current.data(), d_current,
               SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Debug-Ausgabe
    int alive = 0;
    for (auto c : h_current) alive += c;
    std::cout << "Alive cells: " << alive << std::endl;

    cudaFree(d_current);
    cudaFree(d_next);
}
