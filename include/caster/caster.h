#pragma once
#include <cmath>
#include <functional>
#include <vector>
#include "distance.h"
#include "distance_container.h"
using namespace std;

class Caster : public IDistanceContainer {
 public:
  Caster(int n, function<void(double)> onErrorCallback,
         function<void(vector<double2>&)> onPositionsCallback)
      : positions(n) {
    onError = onErrorCallback;
    onPositions = onPositionsCallback;
  };

  virtual void simul_step() = 0;
  virtual void prepare(vector<int>& labels){};
  virtual void finish(){};

  vector<double2> positions;
  function<void(double)> onError;
  function<void(vector<double2>&)> onPositions;
};
