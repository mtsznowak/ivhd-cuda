#pragma once
#include <cmath>
#include <vector>
#include "caster/caster.h"
#include "distance.h"
#include <chrono>
using namespace std::chrono;
using namespace std;

class CasterCPU : public Caster {
 public:
  CasterCPU(int n, function<void(float)> onErr,
            function<void(vector<float2>&)> onPos)
      : Caster(n, onErr, onPos) {}
  ~CasterCPU(){};

  virtual void simul_step() override {
    if (!it++) {
      system_clock::time_point now = system_clock::now();
      startTime = time_point_cast<milliseconds>(now).time_since_epoch().count();
    }

    simul_step_cpu();
  
    system_clock::time_point now = system_clock::now();
    long actTime =
        time_point_cast<milliseconds>(now).time_since_epoch().count();

    if (actTime >= startTime + 150) {  // 8 fps
      onPositions(positions);
      now = system_clock::now();
      startTime = time_point_cast<milliseconds>(now).time_since_epoch().count();
    }
  };

  virtual void simul_step_cpu() = 0;

 protected:
  unsigned it;
  long startTime;
};
