#include <vector>
#include "caster/caster_cpu.h"
#include "distance.h"
using namespace std;

class CasterAdadeltaAsync : public CasterCPU {
 public:
  CasterAdadeltaAsync(int n, function<void(double)> onErr,
                      function<void(vector<double2>&)> onPos)
      : CasterCPU(n, onErr, onPos),
        v(n, {0, 0}),
        f(n, {0, 0}),
        decGrad(n, {0, 0}),
        decDelta(n, {0, 0}) {}
  virtual void simul_step_cpu() override;

 protected:
  vector<double2> v;
  vector<double2> f;
  vector<double2> decGrad;
  vector<double2> decDelta;

 private:
  double2 force(DistElem distance);
  double a_factor = 0.9;
  double b_factor = 0.002;
  double w_random = 0.01;
};
