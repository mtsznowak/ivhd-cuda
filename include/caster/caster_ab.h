#include <vector>
#include "caster/caster.h"
#include "distance.h"
using namespace std;

class CasterAB : public Caster {
 public:
  CasterAB(int n, function<void(float)> onErr) : Caster(n, onErr), v(n), f(n) {}
  virtual void simul_step(bool firstStep) override;

 protected:
  vector<float2> v;
  vector<float2> f;

 private:
  float2 force(DistElem distance);
  float a_factor = 0.990545;
  float b_factor = 0.000200945;
  float w_near = 1;
  float w_random = 0.01;
  float w_far = 1;
};
