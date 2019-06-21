#include <cmath>
#include <vector>
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

class Caster : public IDistanceContainer {
 public:
  Caster(int n) : positions(n) {}
  virtual void simul_step(bool firstStep) = 0;
  void addDistance(DistElem dst) { distances.push_back(dst); };
  void sortHostSamples(vector<int> &labels);

  vector<float2> positions;
  vector<DistElem> distances;

  float2 *d_positions;
  DistElem *d_distances;
  Sample *d_samples;

  bool allocateInitializeDeviceMemory();
  bool copyResultsToHost();

 protected:
  void initializeHelperVectors();
};
