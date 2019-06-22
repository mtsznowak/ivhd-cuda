#include <cuda.h>
#include <chrono>
#include <iostream>
#include <vector>
#include "data.h"
#include "caster/caster_cuda_ab.h"
#include "caster/caster_ab.h"
using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
  /*cerr << "loading dataset" << endl;*/
  Data data;
  int n = data.load_mnist(argv[1]);

  CasterCudaAB c(n);
  Caster& caster = c;
  data.generateNearestDistances(caster, n, argv[2]);
  data.generateRandomDistances(caster, n);

  for (int i = 0; i < n; i++) {
    caster.positions[i].x = rand() % 100000 / 100000.0;
    caster.positions[i].y = rand() % 100000 / 100000.0;
  }

  caster.prepare(data.labelsRef());
  cudaDeviceSynchronize();

  auto now = system_clock::now();
  auto start = time_point_cast<milliseconds>(now).time_since_epoch().count();

  for (int i = 0; i < stoi(argv[3]); i++) {
    caster.simul_step(i == 0 ? true : false);
  }
  cudaDeviceSynchronize();

  now = system_clock::now();
  auto totalTime =
      time_point_cast<milliseconds>(now).time_since_epoch().count() - start;
  cerr << totalTime << endl;

  caster.finish();

  for (int i = 0; i < n; i++) {
    if (i % 10 == 0)
      cout << caster.positions[i].x << " " << caster.positions[i].y << " "
           << data.labels[i] << endl;
  }

  return 0;
}
