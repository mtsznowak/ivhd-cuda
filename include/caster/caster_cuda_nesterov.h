#pragma once
#include "caster/caster_cuda.h"
using namespace std;

class CasterCudaNesterov : public CasterCuda {
 public:
  CasterCudaNesterov(int n, function<void(double)> onErr,
                     function<void(vector<double2>&)> onPos)
      : CasterCuda(n, onErr, onPos) {}

 protected:
  virtual void simul_step_cuda() override;
};
