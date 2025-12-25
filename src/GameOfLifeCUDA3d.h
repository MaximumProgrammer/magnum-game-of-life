/*Markus Gruber
 *Markus.Gruber4@gmx.net
 */

#include <iostream>
#include <vector>

bool gpu_available();

void cuda_init(const size_t NX, const size_t NY, const size_t NZ);

void cuda_free();

void next_step(std::vector<int>& h_next, const std::vector<int>& h_current,
               const size_t NX, const size_t NY, const size_t NZ);
