#include "neuralforge/schema.hpp"
#include <set>
#include <stdexcept>
namespace neuralforge {
namespace {
void names(std::vector<std::string>& values,std::size_t count,const char* prefix) {
 if(values.empty())for(std::size_t i=0;i<count;++i)values.push_back(std::string(prefix)+std::to_string(i));
 if(values.size()!=count)throw std::invalid_argument("Schema name count mismatch");
 std::set<std::string> unique;
 for(const auto& s:values) {
  bool control=false;for(unsigned char c:s)control=control||c<32||c==127;
  if(s.empty()||s.size()>128||control||!unique.insert(s).second)
   throw std::invalid_argument("Schema names must be unique, nonempty, printable and at most 128 bytes");
 }
}
}
void validate_schema(FeatureSchema& s,std::size_t inputs,std::size_t outputs) {
 names(s.features,inputs,"feature_");names(s.classes,outputs,"class_");
}
}
