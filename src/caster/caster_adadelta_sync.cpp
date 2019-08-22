#include "caster/caster_adadelta_sync.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace std;

#define DECAYING_PARAM 0.9
#define EPS 0.00000001

float2 CasterAdadeltaSync::force(DistElem distance) {
  float2 rv = {positions[distance.i].x - positions[distance.j].x,
               positions[distance.i].y - positions[distance.j].y};

  float r = sqrt(rv.x * rv.x + rv.y * rv.y + 0.00001f);
  float D = distance.r;

  float energy = (D - r) / r;

  return {rv.x * energy, rv.y * energy};
}

float2 CasterAdadeltaSync::calcForce(int i) {
  float2 df = {0, 0};
  for (int j = 0; j < neighbours[i].size(); j++) {
    float2 dfcomponent = force(neighbours[i][j]);
    if (neighbours[i][j].type == etRandom) {
      dfcomponent.x *= w_random;
      dfcomponent.y *= w_random;
    }
    if (distances[i].i == i) {
      df = {df.x + dfcomponent.x, df.y + dfcomponent.y};
    } else {
      df = {df.x - dfcomponent.x, df.y - dfcomponent.y};
    }
  }

  return df;
}

void CasterAdadeltaSync::simul_step_cpu() {
  // update velicities and positions
  for (int i = 0; i < positions.size(); i++) {
    float2 force = calcForce(i);

    decGrad[i].x = decGrad[i].x * DECAYING_PARAM +
                   (1.0 - DECAYING_PARAM) * force.x * force.x;
    decGrad[i].y = decGrad[i].y * DECAYING_PARAM +
                   (1.0 - DECAYING_PARAM) * force.y * force.y;

    float deltax =
        force.x / sqrtf(EPS + decGrad[i].x) * sqrtf(EPS + decDelta[i].x);
    float deltay =
        force.y / sqrtf(EPS + decGrad[i].y) * sqrtf(EPS + decDelta[i].y);

    positions[i].x += deltax;
    positions[i].y += deltay;

    decDelta[i].x = decDelta[i].x * DECAYING_PARAM +
                    (1.0 - DECAYING_PARAM) * deltax * deltax;
    decDelta[i].y = decDelta[i].y * DECAYING_PARAM +
                    (1.0 - DECAYING_PARAM) * deltay * deltay;
  }
}

void CasterAdadeltaSync::prepare(vector<int> &labels) {
  // initialize nn array
  for (int i = 0; i < distances.size(); i++) {
    neighbours[distances[i].i].push_back(distances[i]);
    neighbours[distances[i].j].push_back(distances[i]);
  }
}
