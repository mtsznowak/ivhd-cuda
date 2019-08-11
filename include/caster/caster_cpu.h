#pragma once
#include <cmath>
#include <vector>
#include "caster/caster.h"
#include "distance.h"
using namespace std;

class CasterCPU : public Caster {
 public:
  CasterCPU(int n, function<void(float)> onErr, function<void(vector<float2>&)> onPos)
      : Caster(n, onErr, onPos) {}
  ~CasterCPU(){};

  virtual void simul_step() override {
    simul_step_cpu();
    if (it++ % 100 == 0) {
      float err = 0.0;
      for (auto& dist : distances) {
        float d = dist.r;
        float2 iPos = positions[dist.i];
        float2 jPos = positions[dist.j];
        float2 ij = {iPos.x - jPos.x, jPos.y - jPos.y};
        err += abs(d - sqrt(ij.x * ij.x + ij.y * ij.y));
      }
      onError(err);
    }
  };

  virtual void simul_step_cpu() = 0;

 protected:
  unsigned it;
};
