#include <vector>
#include "caster/caster_cpu.h"
#include "distance.h"
using namespace std;

class CasterAB : public CasterCPU {
 public:
  CasterAB(int n, function<void(double)> onErr, function<void(vector<double2>&)> onPos)
      : CasterCPU(n, onErr, onPos), v(n, {0, 0}), f(n, {0, 0}) {}
  virtual void simul_step_cpu() override;

 protected:
  vector<double2> v;
  vector<double2> f;

 private:
  double2 force(DistElem distance);
  double a_factor = 0.9;
  double b_factor = 0.002;
  double w_random = 0.01;
};
