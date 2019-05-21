#include <functional>
#include <string>
using namespace std;

class KNNParser {
 public:
  void parseFile(std::string fileName, function<void(int, int)> handlePair);
};
