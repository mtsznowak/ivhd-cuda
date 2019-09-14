#include "data.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include "csv-parser/parser.hpp"
#include "knn_parser.h"
using namespace std;

int Data::load_mnist(std::string file) {
  ifstream f(file);
  int n, m;

  f >> n >> m;

  f.clear();
  f.seekg(0);
  aria::csv::CsvParser parser(f);

  mnist.resize(n);
  int mnist_ind = 0;
  int parserit = 0;
  for (auto& row : parser) {
    if (parserit++ <= 1) {
      continue;
    }

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

#include <iostream>
void Data::generateRandomDistances(IDistanceContainer& dstContainer, int n,
                                   unsigned rn) {
  for (int i = 0; i < n; i++) {
    unsigned rn_it = rn;
    while (rn_it--) {
      int randIndex = rand() % n;
      while (randIndex == i && !dstContainer.containsDst(i, randIndex)) {
        randIndex = rand() % n;
      }

      DistElem distElem(i, randIndex, DistElemType::etRandom, 1);
      dstContainer.addDistance(distElem);
    }
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
