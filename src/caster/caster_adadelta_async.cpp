#include "caster/caster_adadelta_async.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace std;

#define DECAYING_PARAM 0.9
#define EPS 0.00000001

float2 CasterAdadeltaAsync::force(DistElem distance) {
  float2 rv = {positions[distance.i].x - positions[distance.j].x,
               positions[distance.i].y - positions[distance.j].y};

  float r = sqrt(rv.x * rv.x + rv.y * rv.y + 0.00001f);
  float D = distance.r;

  float energy = (D - r) / r;

  return {rv.x * energy, rv.y * energy};
}

void CasterAdadeltaAsync::simul_step_cpu() {
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
    decGrad[i].x = decGrad[i].x * DECAYING_PARAM +
                   (1.0 - DECAYING_PARAM) * f[i].x * f[i].x;
    decGrad[i].y = decGrad[i].y * DECAYING_PARAM +
                   (1.0 - DECAYING_PARAM) * f[i].y * f[i].y;

    float deltax =
        f[i].x / sqrtf(EPS + decGrad[i].x) * sqrtf(EPS + decDelta[i].x);
    float deltay =
        f[i].y / sqrtf(EPS + decGrad[i].y) * sqrtf(EPS + decDelta[i].y);

    positions[i].x += deltax;
    positions[i].y += deltay;

    decDelta[i].x = decDelta[i].x * DECAYING_PARAM +
                    (1.0 - DECAYING_PARAM) * deltax * deltax;
    decDelta[i].y = decDelta[i].y * DECAYING_PARAM +
                    (1.0 - DECAYING_PARAM) * deltay * deltay;
  }
}
