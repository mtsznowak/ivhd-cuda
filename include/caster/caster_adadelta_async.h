#include <vector>
#include "caster/caster_cpu.h"
#include "distance.h"
using namespace std;

class CasterAdadeltaAsync : public CasterCPU {
 public:
  CasterAdadeltaAsync(int n, function<void(float)> onErr,
                      function<void(vector<float2>&)> onPos)
      : CasterCPU(n, onErr, onPos),
        v(n, {0, 0}),
        f(n, {0, 0}),
        decGrad(n, {0, 0}),
        decDelta(n, {0, 0}) {}
  virtual void simul_step_cpu() override;

 protected:
  vector<float2> v;
  vector<float2> f;
  vector<float2> decGrad;
  vector<float2> decDelta;

 private:
  float2 force(DistElem distance);
  float a_factor = 0.9;
  float b_factor = 0.002;
  float w_random = 0.01;
};
