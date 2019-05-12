#include "from_ivta.h"
#include <cuda.h>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <iostream>
#include "constants.h"
using namespace std;

anyVector3d IVHD::force(DistElem distance) {
  anyVector3d rv = positions[distance.i] - positions[distance.j];

  Real r = positions[distance.i].distance3D(positions[distance.j]);
  Real D = distance.r;

  // if (distance.type == etNear)
  //  D *= shrink_near;
  // else if (distance.type == etFar)
  //  D *= shrink_far;

  Real energy = (r - D) / r;

  return rv * (-energy);
}

__global__ void calcPositions(long n, anyVector3d *v, anyVector3d *f, anyVector3d *positions) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) {
    v[i].x = v[i].x * a_factor + f[i].x * b_factor;
    v[i].y = v[i].y * a_factor + f[i].y * b_factor;
    positions[i].x += v[i].x;
    positions[i].y += v[i].y;
  }
  return;
}

__global__ void calcForceComponents(int compNumber, anyVector3d *components, DistElem *distances, anyVector3d *positions) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < compNumber) {
    DistElem distance = distances[i];
    anyVector3d posI = positions[distance.i];
    anyVector3d posJ = positions[distance.j];

    anyVector3d rv = posI;
    rv.x -= posJ.x;
    rv.y -= posJ.y;

    Real r = sqrtf((posI.x - posJ.x) * (posI.x - posJ.x) + (posI.y - posJ.y) * (posI.y - posJ.y));
    Real D = distance.r;

    Real energy = (r - D) / r;
    rv.x *= -energy;
    rv.y *= -energy;
    
    // distances are sorted by their type 
    if(distance.type == etRandom) {
      rv.x *= w_random;
      rv.y *= w_random;
    }
    components[i] = rv;
  }
  return;
}


__global__ void applyForces(int n, int k, anyVector3d *f, anyVector3d *components) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < n) {
    for (int j=0; j<k; j++) {
      f[i].x += components[i*k + j].x;
      f[i].y += components[j*k + j].y;
    }
    f[i].x += components[n*k + i].x;
    f[i].y += components[n*k + i].y;
  }
  return;
}


void IVHD::time_step_R(bool firstStep) {
  if (firstStep) {
    cudaMemset(gpu_v, 0, v.size()*sizeof(anyVector3d));
  } else {
    calcPositions<<<positions.size()/256 + 1, 256>>>(positions.size(), gpu_v, gpu_f, gpu_positions);
  }

  // calculate forces
  cudaMemset(gpu_f, 0, f.size() * sizeof(anyVector3d));

  calcForceComponents<<<distances.size() / 256 + 1, 256>>>(distances.size(), gpu_components, gpu_distances, gpu_positions);

  applyForces<<<positions.size()/256 + 1, 256>>>(positions.size(), 2, gpu_f, gpu_components);
  /*  f[distances[i].i] += df;*/
  /*  f[distances[i].j] -= df;*/
  /*}*/
}

bool IVHD::allocateInitializeDeviceMemory() {
  cuCall(cudaMalloc(&gpu_positions, positions.size() * sizeof(anyVector3d)));
  cuCall(cudaMalloc(&gpu_v, v.size() * sizeof(anyVector3d)));
  cuCall(cudaMalloc(&gpu_f, f.size() * sizeof(anyVector3d)));
  cuCall(cudaMalloc(&gpu_distances, distances.size() * sizeof(DistElem)));
  cuCall(cudaMalloc(&gpu_components, distances.size() * sizeof(anyVector3d)));

  cuCall(cudaMemcpy(gpu_positions, &positions[0], sizeof(anyVector3d) * positions.size(), cudaMemcpyHostToDevice));
  cuCall(cudaMemcpy(gpu_v, &v[0], sizeof(anyVector3d) * v.size(), cudaMemcpyHostToDevice));
  cuCall(cudaMemcpy(gpu_f, &f[0], sizeof(anyVector3d) * f.size(), cudaMemcpyHostToDevice));
  cuCall(cudaMemcpy(gpu_distances, &distances[0], sizeof(DistElem) * distances.size(), cudaMemcpyHostToDevice));

  return true;
}

bool IVHD::copyResultsToHost() {
  cuCall(cudaMemcpy(&positions[0], gpu_positions, sizeof(anyVector3d) * positions.size(), cudaMemcpyDeviceToHost));

  cuCall(cudaFree(gpu_positions));
  cuCall(cudaFree(gpu_v));
  cuCall(cudaFree(gpu_f));
  cuCall(cudaFree(gpu_distances));
  cuCall(cudaFree(gpu_components));

  return true;
}
