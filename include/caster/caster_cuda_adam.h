#pragma once
#include "caster/caster_cuda.h"
using namespace std;

class CasterCudaAdam : public CasterCuda {
 public:
  CasterCudaAdam(int n, function<void(float)> onErr,
                 function<void(vector<float2>&)> onPos)
      : CasterCuda(n, onErr, onPos) {}
  virtual void prepare(vector<int> &labels) override;

 protected:
  virtual void simul_step_cuda() override;

 private:
  // x,y -> squared mean, z,w -> mean
  float4 *d_average_params;
};
