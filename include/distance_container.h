#pragma once
#include <vector>
#include "distance.h"

class IDistanceContainer {
 public:
  virtual ~IDistanceContainer() {}
  virtual void addDistance(DistElem dst) { distances.push_back(dst); };
  virtual bool containsDst(int i, int j) {
    for (auto &d : distances) {
      return (d.i == i && d.j == j) || (d.i == j && d.j == i);
    }
    return false;
  };

 protected:
  std::vector<DistElem> distances;
};
