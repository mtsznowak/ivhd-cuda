#pragma once
#include "caster/caster_cuda.h"
using namespace std;

class CasterCudaAdadelta : public CasterCuda {
 public:
  CasterCudaAdadelta(int n, function<void(double)> onErr,
                     function<void(vector<double2>&)> onPos)
      : CasterCuda(n, onErr, onPos) {}
  virtual void prepare(vector<int> &labels) override;

 protected:
  virtual void simul_step_cuda() override;

 private:
  // x,y -> gradient, z,w -> param
  double4 *d_average_params;
};
