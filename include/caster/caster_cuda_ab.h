#pragma once
#include "caster/caster_cuda.h"
using namespace std;

class CasterCudaAB : public CasterCuda {
 public:
  CasterCudaAB(int n, function<void(float)> onErr,
               function<void(vector<float2>&)> onPos)
      : CasterCuda(n, onErr, onPos) {}

  void setFinalizing(bool val) { finalizing = val; }

 protected:
  virtual void simul_step_cuda() override;
  bool finalizing = false;
};
