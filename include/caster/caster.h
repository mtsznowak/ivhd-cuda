#pragma once
#include <cmath>
#include <functional>
#include <vector>
#include "distance.h"
#include "distance_container.h"
using namespace std;

class Caster : public IDistanceContainer {
 public:
  Caster(int n, function<void(float)> onErrorCallback) : positions(n) {
    onError = onErrorCallback;
  };

  virtual void simul_step() = 0;
  virtual void prepare(vector<int>& labels){};
  virtual void finish(){};

  vector<float2> positions;
  function<void(float)> onError;
};
