#pragma once
#include "distance.h"

class IDistanceContainer {
 public:
  virtual ~IDistanceContainer() {}
  virtual void addDistance(DistElem distElem) = 0;
};
