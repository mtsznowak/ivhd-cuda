#pragma once
#include "distance.h"
#include <vector>

class IDistanceContainer {
 public:
  virtual ~IDistanceContainer() {}
  virtual void addDistance(DistElem dst) { distances.push_back(dst); };

 protected:
  std::vector<DistElem> distances;
};
