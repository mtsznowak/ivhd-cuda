#pragma once
#include <vector>
#include "distance.h"
#include "distance_container.h"
using namespace std;

class Caster : public IDistanceContainer {
 public:
  Caster(int n) : positions(n) {}
  virtual void simul_step(bool firstStep) = 0;
  virtual void prepare(vector<int> &labels){};
  virtual void finish(){};
  void addDistance(DistElem dst) { distances.push_back(dst); };

  vector<float2> positions;
  vector<DistElem> distances;
};
