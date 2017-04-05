#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <string>
#include <iostream>

namespace fs = boost::filesystem;


int main() {
  fs::path prefix = "/home/hebi/data/fast/WaxPatch-master/WaxPatch/WaxPatch/wax/extensions/json/yajl/yajl_common.h";
  fs::path target = "/home/hebi/data/fast/WaxPatch-master";

  std::cout << fs::relative(prefix, target) << "\n";
}
