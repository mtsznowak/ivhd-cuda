#include "cuda_caster.h"
using namespace std;

class CasterAB : public CudaCaster {
 public:
  CasterAB(int n) : CudaCaster(n) {}
  void simul_step(bool firstStep) override;
};
