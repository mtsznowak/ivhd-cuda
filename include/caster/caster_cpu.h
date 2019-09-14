#pragma once
#include <cmath>
#include <vector>
#include "caster/caster.h"
#include "distance.h"
using namespace std;

class CasterCPU : public Caster {
 public:
  CasterCPU(int n, function<void(double)> onErr, function<void(vector<double2>&)> onPos)
      : Caster(n, onErr, onPos) {}
  ~CasterCPU(){};

  virtual void simul_step() override {
    simul_step_cpu();
    if (it++ % 100 == 0) {
      double err = 0.0;
      for (auto& dist : distances) {
        double d = dist.r;
        double2 iPos = positions[dist.i];
        double2 jPos = positions[dist.j];
        double2 ij = {iPos.x - jPos.x, jPos.y - jPos.y};
        err += abs(d - sqrt(ij.x * ij.x + ij.y * ij.y));
      }
      onError(err);
    }
  };

  virtual void simul_step_cpu() = 0;

 protected:
  unsigned it;
};
