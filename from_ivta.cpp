#include "from_ivta.h"
#include <algorithm>
#include <cstring>
#include <iostream>
using namespace std;

anyVector3d IVHD::force(DistElem distance, Real &energy) {
  anyVector3d rv = positions[distance.i] - positions[distance.j];

  Real r = positions[distance.i].distance3D(positions[distance.j]);
  Real D = distance.r;

  if (distance.type == etNear)
    D *= shrink_near;
  else if (distance.type == etFar)
    D *= shrink_far;

  energy = (r - D) / r;

  return rv * (-energy);
}

void IVHD::time_step_R(bool firstStep, Real &energy, Real & /*dtf*/,
                       long &interactions) {
  if (firstStep) {
    for (int i = 0; i < v.size(); i++) v[i] = anyVector3d(0, 0, 0);
  } else {
    for (int i = 0; i < positions.size(); i++) {
      v[i] = v[i] * a_factor + f[i] * b_factor;
      positions[i] += v[i];

      if (only2d) positions[i].z = 0;
    }
  }

  // calculate forces
  for (int i = 0; i < f.size(); i++) {
    f[i].set(0, 0, 0);
  }

  Real de;
  energy = 0;
  interactions = 0;
  for (int i = 0; i < distances.size(); i++) {
    interactions++;
    anyVector3d df = force(distances[i], de);

    if (distances[i].type == etNear) {
      df = df * w_near;
    } else if (distances[i].type == etRandom) {
      df = df * w_random;
    } else {
      df = df * w_far;
    }

    energy += de * de;
    f[distances[i].i] += df;
    f[distances[i].j] -= df;
  }

  if (interactions) energy /= interactions;
}
