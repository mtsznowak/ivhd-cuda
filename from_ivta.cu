#include <cuda.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include "constants.h"
#include "from_ivta.h"
using namespace std;

__global__ void calcPositions(long n, float2 *v, float2 *f,
                              float2 *positions) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    v[i].x = v[i].x * a_factor + f[i].x * b_factor;
    v[i].y = v[i].y * a_factor + f[i].y * b_factor;
    positions[i].x += v[i].x;
    positions[i].y += v[i].y;
  }
  return;
}

__global__ void calcForceComponents(int compNumber, float2 *components,
                                    DistElem *distances,
                                    float2 *positions) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < compNumber) {
    DistElem distance = distances[i];
    float2 posI = positions[distance.i];
    float2 posJ = positions[distance.j];

    float2 rv = posI;
    rv.x -= posJ.x;
    rv.y -= posJ.y;

    Real r = sqrtf((posI.x - posJ.x) * (posI.x - posJ.x) +
                   (posI.y - posJ.y) * (posI.y - posJ.y));
    Real D = distance.r;

    Real energy = (r - D) / r;
    rv.x *= -energy;
    rv.y *= -energy;

    // distances are sorted by their type
    if (distance.type == etRandom) {
      rv.x *= w_random;
      rv.y *= w_random;
    }
    components[i] = rv;
  }
  return;
}

__global__ void applyForces(int n, float2 *f, DistElem *dstElems,
                            float2 *components, int *lens,
                            int **dst_indexes, int *sample_indexes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    i = sample_indexes[i];
    int dst_len = lens[i];

    for (int j = 0; j < dst_len; j++) {
      int dst_index = dst_indexes[i][j];
      DistElem &dst = dstElems[dst_index];
      int negat = ((i == dst.i) - (i == dst.j));
      float2 &comp = components[dst_index];
      f[i].x += comp.x * negat;
      f[i].y += comp.y * negat;
    }
  }
  return;
}

void IVHD::initializeHelperVectors() {
  // calculate number of distances for each sample
  std::unordered_map<int, int> sampleFreq;
  for (const auto &dst : distances) {
    sampleFreq[dst.i]++;
    sampleFreq[dst.j]++;
  }

  // generate sorted (by the smallest number of distances) list of sample
  // indexes
  vector<int> sampleIndexes;
  for (int i = 0; i < positions.size(); i++) {
    sampleIndexes.push_back(i);
  }

  sort(sampleIndexes.begin(), sampleIndexes.end(),
       [&sampleFreq](const int &a, const int &b) -> bool {
         if (sampleFreq[a] != sampleFreq[b]) {
           return sampleFreq[a] < sampleFreq[b];
         } else {
           return a < b;
         }
       });

  cuCall(cudaMalloc(&gpu_sample_indexes, sizeof(int) * positions.size()));
  cuCall(cudaMemcpy(gpu_sample_indexes, &sampleIndexes[0],
                    sampleIndexes.size() * sizeof(int),
                    cudaMemcpyHostToDevice));

  vector<vector<int>> dst_indexes_vec(positions.size());

  for (int i = 0; i < distances.size(); i++) {
    dst_indexes_vec[distances[i].i].push_back(i);
    dst_indexes_vec[distances[i].j].push_back(i);
  }

  cuCall(cudaMalloc(&gpu_dst_indexes, positions.size() * sizeof(int *)));
  dst_indexes = (int **)malloc(positions.size() * sizeof(int *));

  for (int i = 0; i < positions.size(); i++) {
    cuCall(
        cudaMalloc(&dst_indexes[i], dst_indexes_vec[i].size() * sizeof(int)));
    cuCall(cudaMemcpy(dst_indexes[i], &dst_indexes_vec[i][0],
                      dst_indexes_vec[i].size() * sizeof(int),
                      cudaMemcpyHostToDevice));
  }

  cuCall(cudaMemcpy(gpu_dst_indexes, dst_indexes,
                    positions.size() * sizeof(int *), cudaMemcpyHostToDevice));

  int sizes[positions.size()];
  for (int i = 0; i < positions.size(); i++) {
    sizes[i] = dst_indexes_vec[i].size();
  }
  cuCall(cudaMalloc(&gpu_dst_lens, sizeof(int) * positions.size()));
  cuCall(cudaMemcpy(gpu_dst_lens, sizes, sizeof(int) * positions.size(),
                    cudaMemcpyHostToDevice));
}

void IVHD::time_step_R(bool firstStep) {
  if (firstStep) {
    cudaMemset(gpu_v, 0, v.size() * sizeof(float2));
    initializeHelperVectors();
  } else {
    calcPositions<<<positions.size() / 256 + 1, 256>>>(positions.size(), gpu_v,
                                                       gpu_f, gpu_positions);
  }

  // calculate forces
  cudaMemset(gpu_f, 0, f.size() * sizeof(float2));

  calcForceComponents<<<distances.size() / 256 + 1, 256>>>(
      distances.size(), gpu_components, gpu_distances, gpu_positions);

  // calculate index of every force that should be applied for given sample
  applyForces<<<positions.size() / 256 + 1, 256>>>(
      positions.size(), gpu_f, gpu_distances, gpu_components, gpu_dst_lens,
      gpu_dst_indexes, gpu_sample_indexes);
}

bool IVHD::allocateInitializeDeviceMemory() {
  cuCall(cudaMalloc(&gpu_positions, positions.size() * sizeof(float2)));
  cuCall(cudaMalloc(&gpu_v, v.size() * sizeof(float2)));
  cuCall(cudaMalloc(&gpu_f, f.size() * sizeof(float2)));
  cuCall(cudaMalloc(&gpu_distances, distances.size() * sizeof(DistElem)));
  cuCall(cudaMalloc(&gpu_components, distances.size() * sizeof(float2)));

  cuCall(cudaMemcpy(gpu_positions, &positions[0],
                    sizeof(float2) * positions.size(),
                    cudaMemcpyHostToDevice));
  cuCall(cudaMemcpy(gpu_v, &v[0], sizeof(float2) * v.size(),
                    cudaMemcpyHostToDevice));
  cuCall(cudaMemcpy(gpu_f, &f[0], sizeof(float2) * f.size(),
                    cudaMemcpyHostToDevice));
  cuCall(cudaMemcpy(gpu_distances, &distances[0],
                    sizeof(DistElem) * distances.size(),
                    cudaMemcpyHostToDevice));

  return true;
}

bool IVHD::copyResultsToHost() {
  cuCall(cudaMemcpy(&positions[0], gpu_positions,
                    sizeof(float2) * positions.size(),
                    cudaMemcpyDeviceToHost));

  cuCall(cudaFree(gpu_positions));
  cuCall(cudaFree(gpu_v));
  cuCall(cudaFree(gpu_f));
  cuCall(cudaFree(gpu_distances));
  cuCall(cudaFree(gpu_components));

  return true;
}
