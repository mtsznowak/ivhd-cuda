#pragma once
#include "caster/caster_cuda.h"
using namespace std;

class CasterCudaAB : public CasterCuda {
 public:
  CasterCudaAB(int n, function<void(double)> onErr, function<void(vector<double2>&)> onPos)
      : CasterCuda(n, onErr, onPos) {}

 protected:
  virtual void simul_step_cuda() override;
};
