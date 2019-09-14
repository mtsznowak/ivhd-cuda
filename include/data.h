#include "distance_container.h"
#include "string"
#include "vector"
using namespace std;

class Data {
 public:
  vector<vector<int>> mnist;
  vector<int> labels;

  void generateNearestDistances(IDistanceContainer& dstContainer, int n,
                                string file);
  void generateRandomDistances(IDistanceContainer& dstContainer, int n, unsigned rn);
  int load_mnist(std::string file);
  vector<int>& labelsRef() { return labels; }
};
