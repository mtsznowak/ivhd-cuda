#include "distance_container.h"
#include "string"
#include "vector"
using namespace std;

class Data {
 public:
  vector<vector<int>> mnist;
  vector<int> labels;

  float calcEuclideanDistance(const vector<int>& v1, const vector<int>& v2);
  void generateNearestDistances(IDistanceContainer& dstContainer, int n, string file);
  void generateRandomDistances(IDistanceContainer& dstContainer, int n);
  int load_mnist(std::string file);
};
