#include <cuda.h>
#include "constants.h"
#include "caster/caster_cuda_ab.h"
using namespace std;

__global__ void calcPositions(long n, Sample *samples) {
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

__global__ void calcForceComponents(int compNumber, DistElem *distances,
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

void CasterCudaAB::simul_step_cuda() {
  calcForceComponents<<<256, 256>>>(distances.size(), d_distances, d_samples);
  calcPositions<<<256, 256>>>(positions.size(), d_samples);
}
