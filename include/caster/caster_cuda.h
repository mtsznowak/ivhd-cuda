#pragma once
#include <vector>
#include "caster/caster.h"
#include "distance.h"
#include "distance_container.h"
using namespace std;

struct Sample {
  double2 pos;
  double2 v;
  double2 *components;
  short num_components;
};

class CasterCuda : public Caster {
 public:
  CasterCuda(int n, function<void(double)> onErr, function<void(vector<double2>&)> onPos)
      : Caster(n, onErr, onPos) {}

  void sortHostSamples(vector<int> &labels);

  double2 *d_positions;
  DistElem *d_distances;
  Sample *d_samples;
  double *d_errors;

  virtual void prepare(vector<int> &labels) override;
  virtual void finish() override;
  virtual void simul_step() override;

  bool allocateInitializeDeviceMemory();
  bool copyResultsToHost();

 protected:
  void initializeHelperVectors();
  virtual void simul_step_cuda() = 0;
  virtual double getError();
  void copyPositions();
  int itToPosReady = -1;
  unsigned it = 0;
};
