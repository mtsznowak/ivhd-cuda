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

class IVHD : public IDistanceContainer {
 public:
  IVHD(int n) : positions(n) {}
  void time_step_R(bool firstStep);
  void addDistance(DistElem dst) { distances.push_back(dst); };
  void sortHostSamples(vector<int> &labels);

  vector<float2> positions;
  vector<DistElem> distances;

  float2 *d_positions;
  DistElem *d_distances;
  Sample *d_samples;

  bool allocateInitializeDeviceMemory();
  bool copyResultsToHost();

 private:
  void initializeHelperVectors();
};
