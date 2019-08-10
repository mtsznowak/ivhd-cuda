#include <cuda.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include "caster/caster_ab.h"
#include "caster/caster_adadelta_async.h"
#include "caster/caster_adadelta_sync.h"
#include "caster/caster_cuda_ab.h"
#include "caster/caster_cuda_adadelta.h"
#include "caster/caster_cuda_adam.h"
#include "caster/caster_cuda_nesterov.h"
#include "caster/caster_nesterov.h"
#include "data.h"
using namespace std;
using namespace std::chrono;

string dataset_file;
string knn_file;
string experiment_name;
string algorithm_name;
unsigned iterations;

void parseArg(int argc, char* argv[]) {
  if (argc != 6) {
    cerr << "Expected 5 arguments:\n";
    cerr << "./ivhd dataset_file knn_file iterations experiment_name "
            "algorithm_name\n";
    exit(-1);
  }

  dataset_file = argv[1];
  knn_file = argv[2];
  iterations = stoi(argv[3]);
  experiment_name = argv[4];
  algorithm_name = argv[5];
}

Caster* getCaster(int n, function<void(float)> onError) {
  if (algorithm_name == "ab") {
    return new CasterAB(n, onError);
  } else if (algorithm_name == "nesterov") {
    return new CasterNesterov(n, onError);
  } else if (algorithm_name == "adadelta_sync") {
    return new CasterAdadeltaAsync(n, onError);
  } else if (algorithm_name == "adadelta_async") {
    return new CasterAdadeltaAsync(n, onError);
  } else if (algorithm_name == "cuda_ab") {
    return new CasterCudaAB(n, onError);
  } else if (algorithm_name == "cuda_nesterov") {
    return new CasterCudaNesterov(n, onError);
  } else if (algorithm_name == "cuda_adadelta") {
    return new CasterCudaAdadelta(n, onError);
  } else if (algorithm_name == "cuda_adam") {
    return new CasterCudaAdam(n, onError);

  } else {
    cerr << "Invalid algorithm_name. Expected one of: 'ab', 'cuda_ab', ";
    cerr << "'nesterov', 'cuda_nesterov', 'cuda_adadelta', 'cuda_adam'\n";
    exit(-1);
  }
}

int main(int argc, char* argv[]) {
  parseArg(argc, argv);

  Data data;
  int n = data.load_mnist(dataset_file);

  system_clock::time_point now = system_clock::now();
  long start = time_point_cast<milliseconds>(now).time_since_epoch().count();

  ofstream errFile;
  errFile.open(experiment_name + "_error");
  auto onError = [&](float err) -> void {
    now = system_clock::now();
    auto time =
        time_point_cast<milliseconds>(now).time_since_epoch().count() - start;

    errFile << time << " " << err << endl;
  };

  Caster* casterPtr = getCaster(n, onError);
  Caster& caster = *casterPtr;

  data.generateNearestDistances(caster, n, knn_file);
  data.generateRandomDistances(caster, n);

  for (int i = 0; i < n; i++) {
    caster.positions[i].x = rand() % 100000 / 100000.0;
    caster.positions[i].y = rand() % 100000 / 100000.0;
  }

  caster.prepare(data.labelsRef());
  cudaDeviceSynchronize();

  now = system_clock::now();
  start = time_point_cast<milliseconds>(now).time_since_epoch().count();

  for (int i = 0; i < iterations; i++) {
    caster.simul_step();
  }
  cudaDeviceSynchronize();

  now = system_clock::now();
  auto totalTime =
      time_point_cast<milliseconds>(now).time_since_epoch().count() - start;
  cerr << totalTime << endl;

  caster.finish();

  ofstream results;
  results.open(experiment_name + "_result");
  for (int i = 0; i < n; i++) {
    if (i % 10 == 0)
      results << caster.positions[i].x << " " << caster.positions[i].y << " "
              << data.labels[i] << endl;
  }

  results.close();
  errFile.close();
  return 0;
}
