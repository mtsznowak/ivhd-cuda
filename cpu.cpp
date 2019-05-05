#include <cstdlib>
#include <iostream>
#include <queue>
#include <vector>
#include "from_ivta.h"
#include "third_party/csv-parser/parser.hpp"
using namespace std;

vector<vector<int>> mnist;
vector<int> labels;

float calcEuclideanDistance(const vector<int>& v1, const vector<int>& v2) {
  float distance = 0;
  for (int i = 0; i < v1.size(); i++) {
    distance += (v1[i] - v2[i]) * (v1[i] - v2[i]);
  }
  return sqrt(distance);
}

int load_mnist() {
  std::ifstream f("data/mnist_test_small.csv");

  aria::csv::CsvParser parser(f);

  int n = 0;
  for (auto& row : parser) {
    if (n >= mnist.size()) {
      mnist.resize(n * 2 + 1);
    }
    int i = 0;
    for (string field : row) {
      if (!i++) {
        labels.push_back(stoi(field));
      } else {
        mnist[n].push_back(stoi(field));
      }
    }
    n++;
  }
  mnist.resize(n);
  return n;
}

void calcDistances(IVHD& ivhd, int n) {
  for (int i = 0; i < n; i++) {
    priority_queue<pair<float, int>> neighboringIndexes;
    for (int j = 0; j < n; j++) {
      if (i == j) {
        continue;
      }
      float ijDistance = calcEuclideanDistance(mnist[i], mnist[j]);
      neighboringIndexes.push({-ijDistance, j});
    }
    int k = 3;
    while (k--) {
      auto top = neighboringIndexes.top();
      neighboringIndexes.pop();
      DistElem distElem;
      distElem.i = i;
      distElem.j = top.second;
      distElem.r = -top.first;
      distElem.type = DistElemType::etNear;
      ivhd.distances.push_back(distElem);
    }

    int randIndex = rand() % n;
    while (randIndex == i) {
      randIndex = rand() % n;
    }
    DistElem distElem;
    distElem.i = i;
    distElem.j = randIndex;
    distElem.r = calcEuclideanDistance(mnist[i], mnist[randIndex]);
    distElem.type = DistElemType::etRandom;
    ivhd.distances.push_back(distElem);
  }
}

int main(int argc, char* argv[]) {
  int n = load_mnist();

  IVHD ivhd(n);

  calcDistances(ivhd, n);

  for (int i = 0; i < n; i++) {
    ivhd.positions[i].x = rand() % 100000 - 50000;
    ivhd.positions[i].y = rand() % 100000 - 50000;
    ivhd.positions[i].z = rand() % 100000 - 50000;
  }

  for (int i = 0; i < 50000; i++) {
    float energy;
    float dtf;
    long interactions;
    ivhd.time_step_R(i == 0 ? true : false, energy, dtf, interactions);
  }

  for (int i = 0; i < n; i++) {
    cout << ivhd.positions[i].x << " " << ivhd.positions[i].y << " "
         << labels[i] << endl;
  }

  return 0;
}
