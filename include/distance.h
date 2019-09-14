#pragma once
#include <cuda_runtime.h>

enum DistElemType { etNear, etFar, etRandom, etToRemove };

class DistElem {
 public:
  long i, j;
  double r;
  double2 *comp1;
  double2 *comp2;
  DistElemType type;

  DistElem(long pi, long pj) : i(pi), j(pj), r(0), type(etNear){};
  DistElem(long pi, long pj, DistElemType ptype, double pr)
      : i(pi), j(pj), r(pr), type(ptype){};
};
