#pragma once
#include "caster/caster_cuda.h"
using namespace std;

class CasterCudaNesterov : public CasterCuda {
 public:
  CasterCudaNesterov(int n, function<void(float)> onErr)
      : CasterCuda(n, onErr) {}

 protected:
  virtual void simul_step_cuda() override;
};
