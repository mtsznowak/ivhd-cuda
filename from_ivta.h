#include <cmath>
#include <vector>
using namespace std;
typedef float Real;

enum DistElemType { etNear, etFar, etRandom, etToRemove };

class DistElem {
 public:
  long i, j;
  Real r;
  DistElemType type;
};

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

class IVHD {
 public:
  IVHD(int n) : positions(n), v(n), f(n) {}
  void time_step_R(bool firstStep, Real &energy, Real & /*dtf*/,
                   long &interactions);
  void addDistance(DistElem dst){distances.push_back(dst);};
  vector<anyVector3d> positions;
  vector<anyVector3d> v;
  vector<anyVector3d> f;
  vector<DistElem> distances;

 private:
  anyVector3d force(DistElem distance, Real &energy);
  Real shrink_near = 0, shrink_far = 1;
  Real sammon_k = 1;
  Real sammon_m = 2;
  Real sammon_w = 0;
  Real a_factor = 0.990545;
  Real b_factor = 0.000200945;
  bool only2d = true;
  Real w_near = 1;
  Real w_random = 0.01;
  Real w_far = 1;
};
