#pragma once
#include "neuralforge/model.hpp"
#include "neuralforge/adaptation.hpp"
#include <random>
namespace nf_test {
inline neuralforge::MLPModel model(std::size_t in=2,std::size_t hidden=9,std::size_t out=3) {
 std::array<neuralforge::Tensor,4> p{neuralforge::Tensor({in,hidden}),neuralforge::Tensor({hidden}),
  neuralforge::Tensor({hidden,out}),neuralforge::Tensor({out})};
 std::mt19937 rng(42);std::uniform_real_distribution<float> values(-0.2f,0.2f);
 for(auto& t:p)for(std::size_t i=0;i<t.size();++i)t.at(i)=values(rng);
 return neuralforge::MLPModel::from_parameters(p);
}
inline neuralforge::MLPModel identity(bool swapped=false) {
 std::array<neuralforge::Tensor,4> p{neuralforge::Tensor{{1,0},{0,1}},neuralforge::Tensor({2},0),
  swapped?neuralforge::Tensor{{0,1},{1,0}}:neuralforge::Tensor{{1,0},{0,1}},neuralforge::Tensor({2},0)};
 return neuralforge::MLPModel::from_parameters(p);
}
inline neuralforge::LabeledBatch cohort(bool swapped=false,std::size_t rows=20) {
 neuralforge::LabeledBatch data{neuralforge::Tensor({rows,2}),{}};
 for(std::size_t i=0;i<rows;++i) {
  const auto c=i%2;data.inputs.at(i,0)=c?1:2;data.inputs.at(i,1)=c?2:1;
  data.labels.push_back(swapped?1-c:c);
 }
 return data;
}
}
