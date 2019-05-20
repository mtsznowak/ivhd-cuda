#include <cuda.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include "constants.h"
#include "from_ivta.h"
using namespace std;

__global__ void calcPositions(long n, Sample *samples) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    Sample sample = samples[i];
    sample.v.x = sample.v.x * a_factor + sample.f.x * b_factor;
    sample.v.y = sample.v.y * a_factor + sample.f.y * b_factor;
    sample.f = {0, 0};
    sample.pos.x += sample.v.x;
    sample.pos.y += sample.v.y;
    samples[i] = sample;
  }
  return;
}

__global__ void calcForceComponents(int compNumber, DistElem *distances, Sample *samples) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < compNumber) {
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

__global__ void applyForces(int n, Sample *samples) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    Sample sample = samples[i];

    for (int j = 0; j < sample.num_components; j++) {
      sample.f.x += sample.components[j].x;
      sample.f.y += sample.components[j].y;
    }

    samples[i] = sample;
  }
  return;
}

  // initialize pos in Samples
  // initialize num_components
__global__ void initializeSamples(int n, Sample *samples, float2 *positions, short *sampleFreq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    Sample sample;
    sample.pos = positions[i];
    sample.v = sample.f = {0, 0};
    sample.num_components = sampleFreq[i];
    // FIXME - malloc can return NULL
    sample.components = (float2 *)malloc(sample.num_components * sizeof(float2));
    samples[i] = sample; 
  }
}

__global__ void initializeDistances(int nDst, DistElem *distances, short2 *dstIndexes, Sample *samples) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<nDst) {
    DistElem dst = distances[i];
    dst.comp1 = &samples[dst.i].components[dstIndexes[i].x];
    dst.comp2 = &samples[dst.j].components[dstIndexes[i].y];
    distances[i] = dst;
  }
}

void IVHD::initializeHelperVectors() {
  /*
   * calculate number of distances for each sample and index of each distance
   * for a given sample
   */
  short sampleFreq[positions.size()];
  for (int i = 0; i < positions.size(); i++) {
    sampleFreq[i] = 0;
  }

  short2 dstIndexes[distances.size()];

  for (int i = 0; i < distances.size(); i++) {
    dstIndexes[i] = {sampleFreq[distances[i].i]++,
                     sampleFreq[distances[i].j]++};
  }

  // initialize samples
  short *d_sample_freq;
  cuCall(cudaMalloc(&d_sample_freq, positions.size() * sizeof(short)));
  cuCall(cudaMemcpy(d_sample_freq, sampleFreq, sizeof(short) * positions.size(),
                    cudaMemcpyHostToDevice));

  initializeSamples<<<positions.size() / 256 + 1, 256>>>(
      positions.size(), d_samples, d_positions, d_sample_freq);
  cuCall(cudaFree(d_sample_freq));

  // initialize comps in Distances in device memory
  short2 *d_dst_indexes;
  cuCall(cudaMalloc(&d_dst_indexes, distances.size() * sizeof(short2)));
  cuCall(cudaMemcpy(d_dst_indexes, dstIndexes,
                    sizeof(short2) * distances.size(), cudaMemcpyHostToDevice));

  initializeDistances<<<distances.size() / 256 + 1, 256>>>(
      distances.size(), d_distances, d_dst_indexes, d_samples);
  cuCall(cudaFree(d_dst_indexes));
}

/*
 * This function performs the preprocessing on the CPU that is optional
 *
 * Sorts samples by number of their distances and sorts distances by
 * index i or j to utilize cache better. After sorting samples, their indexes
 * change so we have to update distances once more
 */
void IVHD::sortHostSamples(vector<int> &labels) {
  // create array of sorted indexes
  vector<short> sampleFreq(positions.size());
  for (int i = 0; i < positions.size(); i++) {
    sampleFreq[i] = 0;
  }

  vector<int> sampleIndexes(positions.size());
  for (int i = 0; i < positions.size(); i++) {
    sampleIndexes[i] = i;
  }

  sort(sampleIndexes.begin(), sampleIndexes.end(),
       [&sampleFreq](const int &a, const int &b) -> bool {
         if (sampleFreq[a] != sampleFreq[b]) {
           return sampleFreq[a] < sampleFreq[b];
         } else {
           return a < b;
         }
       });

  // create mapping index->new index
  vector<int> newIndexes(positions.size());
  for (int i = 0; i < positions.size(); i++) {
    newIndexes[sampleIndexes[i]] = i;
  }

  // sort positions
  vector<float2> positionsCopy = positions;
  vector<int> labelsCopy = labels;
  for (int i = 0; i < positions.size(); i++) {
    positions[i] = positionsCopy[sampleIndexes[i]];
    labels[i] = labelsCopy[sampleIndexes[i]];
  }

  // update indexes in distances
  for (int i = 0; i < distances.size(); i++) {
    distances[i].i = newIndexes[distances[i].i];
    distances[i].j = newIndexes[distances[i].j];
  }

  // sort distances
  sort(distances.begin(), distances.end(),
       [](const DistElem &a, const DistElem &b) -> bool {
         if (a.i != b.i) {
           return a.i < b.i;
         } else {
           return a.j <= b.j;
         }
       });
}

void IVHD::time_step_R(bool firstStep) {
  if (firstStep) {
    initializeHelperVectors();
  } else {
    calcPositions<<<positions.size() / 256 + 1, 256>>>(positions.size(), d_samples);
  }

  // calculate forces
  calcForceComponents<<<distances.size() / 256 + 1, 256>>>(
      distances.size(), d_distances, d_samples);

  // calculate index of every force that should be applied for given sample
  applyForces<<<positions.size() / 256 + 1, 256>>>(
      positions.size(), d_samples);
}

bool IVHD::allocateInitializeDeviceMemory() {
  cuCall(cudaMalloc(&d_positions, positions.size() * sizeof(float2)));
  cuCall(cudaMalloc(&d_samples, positions.size() * sizeof(Sample)));
  cuCall(cudaMalloc(&d_distances, distances.size() * sizeof(DistElem)));

  cuCall(cudaMemcpy(d_positions, &positions[0],
                    sizeof(float2) * positions.size(), cudaMemcpyHostToDevice));
  cuCall(cudaMemset(d_samples, 0, positions.size() * sizeof(Sample)));
  cuCall(cudaMemcpy(d_distances, &distances[0],
                    sizeof(DistElem) * distances.size(),
                    cudaMemcpyHostToDevice));

  return true;
}

__global__ void copyPosRelease(int N, Sample *samples, float2 *positions) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<N) {
    positions[i] = samples[i].pos;
    free(samples[i].components);
  }
}

bool IVHD::copyResultsToHost() {
  copyPosRelease<<<positions.size() / 256 + 1, 256>>>(positions.size(), d_samples, d_positions);
  cuCall(cudaMemcpy(&positions[0], d_positions,
                    sizeof(float2) * positions.size(), cudaMemcpyDeviceToHost));
  cuCall(cudaFree(d_positions));
  cuCall(cudaFree(d_distances));
  cuCall(cudaFree(d_samples));

  return true;
}
