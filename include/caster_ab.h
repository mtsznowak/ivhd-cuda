#include "caster.h"
using namespace std;

class CasterAB : public Caster {
 public:
  CasterAB(int n) : Caster(n) {}
  void simul_step(bool firstStep) override;
};
