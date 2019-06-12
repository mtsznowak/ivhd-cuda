#include "data.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include "knn_parser.h"
#include "third_party/csv-parser/parser.hpp"
using namespace std;

float Data::calcEuclideanDistance(const vector<int>& v1,
                                  const vector<int>& v2) {
  float distance = 0;
  for (int i = 0; i < v1.size(); i++) {
    distance += (v1[i] - v2[i]) * (v1[i] - v2[i]);
  }
  return sqrt(distance);
}

int Data::load_mnist(std::string file) {
  ifstream f(file);
  int n, m;

  f >> n >> m;
  aria::csv::CsvParser parser(f);

  mnist.resize(n);
  int mnist_ind = 0;
  for (auto& row : parser) {
    int i = 0;
    for (string field : row) {
      if (i++ == m) {
        labels.push_back(stoi(field));
      } else {
        mnist[mnist_ind].push_back(stoi(field));
      }
    }
    mnist_ind++;
  }
  f.close();
  return n;
}

void Data::generateRandomDistances(IDistanceContainer& dstContainer, int n) {
  for (int i = 0; i < n; i++) {
    int randIndex = rand() % n;
    while (randIndex == i) {
      randIndex = rand() % n;
    }

    DistElem distElem(i, randIndex, DistElemType::etRandom,
                      calcEuclideanDistance(mnist[i], mnist[randIndex]));
    dstContainer.addDistance(distElem);
  }
}

void Data::generateNearestDistances(IDistanceContainer& dstContainer, int n,
                                    string file) {
  KNNParser parser;
  auto lambda = [&dstContainer](int x, int y) {
    DistElem distElem(x, y);
    dstContainer.addDistance(distElem);
  };
  parser.parseFile(file, lambda);
}
