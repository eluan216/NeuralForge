#pragma once
#include <cstddef>
#include <string>
#include <vector>
namespace neuralforge {
struct FeatureSchema {std::vector<std::string> features,classes;};
/// Fill omitted names with deterministic defaults; validate explicit names.
void validate_schema(FeatureSchema& schema,std::size_t inputs,std::size_t outputs);
}
