#include <chrono>
#include <iostream>
#include <vector>
#include <cuda.h>
#include "data.h"
#include "from_ivta.h"
using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
  /*cerr << "loading dataset" << endl;*/
  Data data;
  int n = data.load_mnist(argv[1]);

  IVHD ivhd(n);
  data.generateNearestDistances(ivhd, n, argv[2]);
  data.generateRandomDistances(ivhd, n);

  for (int i = 0; i < n; i++) {
    ivhd.positions[i].x = rand() % 100000 - 50000;
    ivhd.positions[i].y = rand() % 100000 - 50000;
    ivhd.positions[i].z = 0;
  }

  /*cerr << "starting IVHD" << endl;*/
  ivhd.allocateInitializeDeviceMemory();
  cudaDeviceSynchronize();

  auto now = system_clock::now();
  auto start = time_point_cast<milliseconds>(now).time_since_epoch().count();

  for (int i = 0; i < stoi(argv[3]); i++) {
    ivhd.time_step_R(i == 0 ? true : false);
  }
  cudaDeviceSynchronize();

  now = system_clock::now();
  auto totalTime =
    time_point_cast<milliseconds>(now).time_since_epoch().count() - start;


  ivhd.copyResultsToHost();
  /*cerr << "IVHD complete (" << totalTime << " ms)" << endl;*/
  cerr  << totalTime << endl;

  for (int i = 0; i < n; i++) {
    if (i % 10 == 0)
      cout << ivhd.positions[i].x << " " << ivhd.positions[i].y << " "
        << data.labels[i] << endl;
  }

  return 0;
}
