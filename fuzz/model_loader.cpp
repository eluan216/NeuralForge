#include "neuralforge/model.hpp"
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,std::size_t size) {
 try {
  std::istringstream input(std::string(reinterpret_cast<const char*>(data),size));
  const auto model=neuralforge::MLPModel::read(input);
  auto workspace=model.prepare();
  std::vector<float> x(model.input_features(),0),y(model.output_features());
  model.predict_proba_into(x,1,y,workspace);
 }catch(const std::exception&) {}
 return 0;
}
