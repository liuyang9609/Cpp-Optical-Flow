#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <unordered_map>

int main(){
  std::unordered_map<std::string, double> parameters;
  parameters.insert(std::make_pair<std::string, double>("alpha", 2.3));
  parameters.insert(std::make_pair<std::string, double>("beta", 1.3));

  cv::FileStorage test("test.xml", cv::FileStorage::READ);
  
  cv::FileNode p = test["parameters"];
  cv::FileNodeIterator it = p.begin(), it_end = p.end();

  for ( ; it != it_end; ++it) {
    std::cout << (*it).name() << " " << (double)(*it)["aa"] << std::endl;
  }
  
  /*test << "parameters" << "{";
  for (auto i:parameters){
    test << i.first << "{" << "asdf" << 1 << "aa" << i.second << "}";
  }
  test << "}";*/
  test.release();
  return 0;
}
