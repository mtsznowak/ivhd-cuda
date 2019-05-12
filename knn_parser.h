#include <string>
#include <functional>
using namespace std;

class KNNParser {
 public:
  void parseFile(std::string fileName, function< void(int, int) > handlePair);
};
