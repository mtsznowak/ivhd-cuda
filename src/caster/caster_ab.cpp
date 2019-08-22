#include "caster/caster_ab.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace std;

float2 CasterAB::force(DistElem distance) {
  float2 rv = {positions[distance.i].x - positions[distance.j].x,
               positions[distance.i].y - positions[distance.j].y};

  float r = sqrt(rv.x * rv.x + rv.y * rv.y + 0.00001f);
  float D = distance.r;

  float energy = (D - r) / r;

  return {rv.x * energy, rv.y * energy};
}

void CasterAB::simul_step_cpu() {
  // calculate forces
  for (int i = 0; i < f.size(); i++) {
    f[i] = {0, 0};
  }

  for (int i = 0; i < distances.size(); i++) {
    float2 df = force(distances[i]);

    if (distances[i].type == etRandom) {
      df.x *= w_random;
      df.y *= w_random;
    }

    f[distances[i].i].x += df.x;
    f[distances[i].i].y += df.y;
    f[distances[i].j].x -= df.x;
    f[distances[i].j].y -= df.y;
  }

  // update velicities and positions
  for (int i = 0; i < positions.size(); i++) {
    v[i].x = v[i].x * a_factor + f[i].x * b_factor;
    v[i].y = v[i].y * a_factor + f[i].y * b_factor;
    positions[i].x += v[i].x;
    positions[i].y += v[i].y;
  }
}
