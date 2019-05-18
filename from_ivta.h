#include <cmath>
#include <vector>
#include "distance.h"
#include "distance_container.h"
using namespace std;

class IVHD : public IDistanceContainer {
 public:
  IVHD(int n) : positions(n), v(n), f(n) {}
  void time_step_R(bool firstStep);
  void addDistance(DistElem dst) { distances.push_back(dst); };
  vector<float2> positions;
  vector<float2> v;
  vector<float2> f;
  vector<DistElem> distances;

  float2 *gpu_positions;
  float2 *gpu_v;
  
  float2 *gpu_f;
  DistElem *gpu_distances;
  float2 *gpu_components;
  int **gpu_dst_indexes;
  int **dst_indexes;
  int *gpu_dst_lens;
  int *gpu_sample_indexes;

  bool allocateInitializeDeviceMemory();
  bool copyResultsToHost();

 private:
  void initializeHelperVectors();
};
