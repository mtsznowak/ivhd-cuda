#pragma once
enum DistElemType { etNear, etFar, etRandom, etToRemove };

class DistElem {
 public:
  long i, j;
  float r;
  DistElem(long pi, long pj) : i(pi), j(pj), r(0), type(etNear){};
  DistElem(long pi, long pj, DistElemType ptype, float pr)
      : i(pi), j(pj), r(pr), type(ptype){};
  DistElemType type;
};


