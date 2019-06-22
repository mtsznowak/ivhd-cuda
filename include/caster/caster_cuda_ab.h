#include "caster/caster_cuda.h"
using namespace std;

class CasterCudaAB : public CasterCuda {
 public:
  CasterCudaAB(int n) : CasterCuda(n) {}
  void simul_step(bool firstStep) override;
};
