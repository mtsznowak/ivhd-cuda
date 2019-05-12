#include <cmath>
#include <vector>
#include "distance.h"
#include "distance_container.h"
using namespace std;
typedef float Real;

class anyVector3d {
 public:
  anyVector3d() : x(0), y(0), z(0) {}
  anyVector3d(Real x, Real y, Real z) : x(x), y(y), z(z) {}
  Real x, y, z;

  Real distance3D(anyVector3d &v) {
    return sqrt((x - v.x) * (x - v.x) + (y - v.y) * (y - v.y) +
                (z - v.z) * (z - v.z));
  }
  anyVector3d operator+(anyVector3d const &v) const {
    return anyVector3d(x + v.x, y + v.y, z + v.z);
  }

  anyVector3d operator-(anyVector3d const &v) const {
    return anyVector3d(x - v.x, y - v.y, z - v.z);
  }

  void operator+=(anyVector3d const &v) {
    x += v.x;
    y += v.y;
    z += v.z;
  }

  void operator-=(anyVector3d const &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
  }

  anyVector3d operator*(Real n) const {
    return anyVector3d(n * x, n * y, n * z);
  }

  void set(Real new_x, Real new_y, Real new_z) {
    x = new_x;
    y = new_y;
    z = new_z;
  }
};

class IVHD : public IDistanceContainer {
 public:
  IVHD(int n) : positions(n), v(n), f(n) {}
  void time_step_R(bool firstStep);
  void addDistance(DistElem dst) { distances.push_back(dst); };
  vector<anyVector3d> positions;
  vector<anyVector3d> v;
  vector<anyVector3d> f;
  vector<DistElem> distances;

  anyVector3d *gpu_positions;
  anyVector3d *gpu_v;
  anyVector3d *gpu_f;
  DistElem *gpu_distances;
  anyVector3d *gpu_components;

  bool allocateInitializeDeviceMemory();
  bool copyResultsToHost();


 private:
  anyVector3d force(DistElem distance);
};
