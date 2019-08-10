#include <cuda.h>
#include "constants.h"
#include "caster/caster_cuda_adam.h"
using namespace std;

#define B1 0.9
#define B2 0.999
#define EPS 0.00000001f
#define LEARNING_RATE 0.002f

__global__ void calcPositionsAdam(long n, Sample *samples, float4 *avarage_params) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
      i += blockDim.x * gridDim.x) {
    Sample sample = samples[i];
    float4 avarage_param = avarage_params[i];

    float2 force = {0, 0};
    for (int j = 0; j < sample.num_components; j++) {
      force.x += sample.components[j].x;
      force.y += sample.components[j].y;
    }

    avarage_param.x = avarage_param.x * B2 + (1.0 - B2) * force.x * force.x;
    avarage_param.y = avarage_param.y * B2 + (1.0 - B2) * force.y * force.y;

    avarage_param.z = avarage_param.z * B1 + (1.0 - B1) * force.x;
    avarage_param.w = avarage_param.w * B1 + (1.0 - B1) * force.y;

    float deltax = LEARNING_RATE * (avarage_param.z / (1.0 - B1)) / (EPS + sqrtf(avarage_param.x / (1.0 - B2)));
    float deltay = LEARNING_RATE * (avarage_param.w / (1.0 - B1)) / (EPS + sqrtf(avarage_param.y / (1.0 - B2)));

    sample.pos.x += deltax;
    sample.pos.y += deltay;

    samples[i] = sample;
    avarage_params[i] = avarage_param;
  }
  return;
}

__global__ void calcForceComponentsAdam(int compNumber, DistElem *distances,
    Sample *samples) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < compNumber;
      i += blockDim.x * gridDim.x) {
    DistElem distance = distances[i];

    float2 posI = samples[distance.i].pos;
    float2 posJ = samples[distance.j].pos;

    float2 rv = posI;
    rv.x -= posJ.x;
    rv.y -= posJ.y;

    float r = sqrtf((posI.x - posJ.x) * (posI.x - posJ.x) +
        (posI.y - posJ.y) * (posI.y - posJ.y));
    float D = distance.r;

    float energy = (r - D) / r;
    rv.x *= -energy;
    rv.y *= -energy;

    // distances are sorted by their type
    if (distance.type == etRandom) {
      rv.x *= w_random;
      rv.y *= w_random;
    }
    *distance.comp1 = rv;
    *distance.comp2 = {-rv.x, -rv.y};
  }
  return;
}

void CasterCudaAdam::simul_step_cuda() {
  calcForceComponentsAdam<<<256, 256>>>(distances.size(), d_distances, d_samples);
  calcPositionsAdam<<<256, 256>>>(positions.size(), d_samples, d_average_params);
}

void CasterCudaAdam::prepare(vector<int> &labels) {
  CasterCuda::prepare(labels);

  // TODO release this memory
  cuCall(cudaMalloc(&d_average_params, positions.size() * sizeof(float4)));
  cuCall(cudaMemset(d_average_params, 0, positions.size() * sizeof(float4)));
}
