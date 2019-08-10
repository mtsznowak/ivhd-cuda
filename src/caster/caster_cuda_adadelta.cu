#include <cuda.h>
#include "constants.h"
#include "caster/caster_cuda_adadelta.h"
using namespace std;

#define DECAYING_PARAM 0.1
#define EPS 0.00000001f

__global__ void calcPositionsAdadelta(long n, Sample *samples, float4 *avarage_params) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
      i += blockDim.x * gridDim.x) {
    Sample sample = samples[i];
    float4 avarage_param = avarage_params[i];

    float2 force = {0, 0};
    for (int j = 0; j < sample.num_components; j++) {
      force.x += sample.components[j].x;
      force.y += sample.components[j].y;
    }

    avarage_param.x = avarage_param.x * DECAYING_PARAM + (1.0 - DECAYING_PARAM) * force.x * force.x;
    avarage_param.y = avarage_param.y * DECAYING_PARAM + (1.0 - DECAYING_PARAM) * force.y * force.y;

    float deltax = force.x / sqrtf(EPS + avarage_param.x) * sqrtf(EPS + avarage_param.z);
    float deltay = force.y / sqrtf(EPS + avarage_param.y) * sqrtf(EPS + avarage_param.w);

    sample.pos.x += deltax;
    sample.pos.y += deltay;

    avarage_param.z = avarage_param.z * DECAYING_PARAM + (1.0 - DECAYING_PARAM) * deltax * deltax;
    avarage_param.w = avarage_param.w * DECAYING_PARAM + (1.0 - DECAYING_PARAM) * deltay * deltay;

    samples[i] = sample;
    avarage_params[i] = avarage_param;
  }
  return;
}

__global__ void calcForceComponentsAdadelta(int compNumber, DistElem *distances,
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

void CasterCudaAdadelta::simul_step_cuda() {
  calcForceComponentsAdadelta<<<64, 96>>>(distances.size(), d_distances, d_samples);
  calcPositionsAdadelta<<<64, 96>>>(positions.size(), d_samples, d_average_params);
}

void CasterCudaAdadelta::prepare(vector<int> &labels) {
  CasterCuda::prepare(labels);

  // TODO release this memory
  cuCall(cudaMalloc(&d_average_params, positions.size() * sizeof(float4)));
  cuCall(cudaMemset(d_average_params, 0, positions.size() * sizeof(float4)));
}
