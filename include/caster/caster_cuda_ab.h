#include "caster/caster_cuda.h"
using namespace std;

class CasterCudaAB : public CasterCuda {
 public:
  CasterCudaAB(int n, function<void(float)> onErr) : CasterCuda(n, onErr) {}
  void simul_step(bool firstStep) override;
};
