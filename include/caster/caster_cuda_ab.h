#include "caster/caster_cuda.h"
using namespace std;

class CasterAB : public CasterCuda {
 public:
  CasterAB(int n) : CasterCuda(n) {}
  void simul_step(bool firstStep) override;
};
