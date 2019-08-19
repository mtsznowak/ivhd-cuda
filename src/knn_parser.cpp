#include "knn_parser.h"
#include <boost/algorithm/string.hpp>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

void KNNParser::parseFile(std::string fileName,
                          function<void(int, int)> handlePair) {
  std::ifstream ifstream(fileName);

  // read header
  long firstLineMaxSize = 64;
  char firstLine[64];
  ifstream.getline(firstLine, firstLineMaxSize);

  std::vector<std::string> splits;
  boost::split(splits, firstLine, [](char c) { return c == ';'; });

  assert(splits.size() == 3);

  int N = stoi(splits[0]);
  int neighbours = stoi(splits[1]);
  int longSize = stoi(splits[2]);

  // fprintf(stderr, "parsing kNN - N=%d neighbours=%d longSize=%d\n", N,
  // neighbours, longSize);

  // byte order test
  long testNum;
  ifstream.read((char *)&testNum, longSize);
  assert(testNum == 0x01020304);

  // parse neighbours
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < neighbours; j++) {
      long neighbour;
      ifstream.read((char *)&neighbour, longSize);
      assert(ifstream.gcount() == longSize);
      handlePair(i, neighbour);
    }
  }

  ifstream.close();
}
