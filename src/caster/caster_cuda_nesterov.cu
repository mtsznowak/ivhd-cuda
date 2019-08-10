#include <cuda.h>
#include "constants.h"
#include "caster/caster_cuda_nesterov.h"
using namespace std;

__global__ void calcPositionsNesterov(long n, Sample *samples) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    Sample sample = samples[i];

    float2 force = {0, 0};
    for (int j = 0; j < sample.num_components; j++) {
      force.x += sample.components[j].x;
      force.y += sample.components[j].y;
    }

    sample.v.x = sample.v.x * a_factor + force.x * b_factor;
    sample.v.y = sample.v.y * a_factor + force.y * b_factor;

    sample.pos.x += sample.v.x;
    sample.pos.y += sample.v.y;

    samples[i] = sample;
  }
  return;
}

__global__ void calcForceComponentsNesterov(int compNumber, DistElem *distances,
                                    Sample *samples) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < compNumber;
       i += blockDim.x * gridDim.x) {
    DistElem distance = distances[i];

    Sample sampleI = samples[distance.i];
    Sample sampleJ = samples[distance.j];
    float2 posI = sampleI.pos;
    float2 posJ = sampleJ.pos;

    // approximate next positions
    posI.x += sampleI.v.x;
    posI.y += sampleI.v.y;
    posJ.x += sampleJ.v.x;
    posJ.y += sampleJ.v.y;

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

void CasterCudaNesterov::simul_step_cuda() {
  calcForceComponentsNesterov<<<64, 96>>>(distances.size(), d_distances, d_samples);
  calcPositionsNesterov<<<64, 96>>>(positions.size(), d_samples);
}
