#pragma once
#include <vector>
#include "caster/caster.h"
#include "distance.h"
#include "distance_container.h"
using namespace std;

struct Sample {
  float2 pos;
  float2 v;
  float2 f;
  float2 *components;
  short num_components;
};

class CasterCuda : public Caster {
 public:
  CasterCuda(int n, function<void(float)> onErr) : Caster(n, onErr) {}

  void sortHostSamples(vector<int> &labels);

  float2 *d_positions;
  DistElem *d_distances;
  Sample *d_samples;
  float *d_errors;

  virtual void prepare(vector<int> &labels) override;
  virtual void finish() override;
  virtual void simul_step() override;

  bool allocateInitializeDeviceMemory();
  bool copyResultsToHost();

 protected:
  void initializeHelperVectors();
  virtual void simul_step_cuda() = 0;
  virtual float getError();

 private:
  unsigned it = 0;
};
