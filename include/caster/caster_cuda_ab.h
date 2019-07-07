#include "caster/caster_cuda.h"
using namespace std;

class CasterCudaAB : public CasterCuda {
 public:
  CasterCudaAB(int n, function<void(float)> onErr) : CasterCuda(n, onErr) {}

 protected:
  virtual void simul_step_cuda() override;
};
