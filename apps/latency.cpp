#include "neuralforge/model.hpp"
#include "neuralforge/kernels.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace neuralforge;
int main(int argc,char** argv) {
 try {
  if(argc!=2)throw std::invalid_argument("Usage: neuralforge_latency model.nfmodel");
  const auto model=MLPModel::load(argv[1]);auto workspace=model.prepare();
  Tensor input({1,model.input_features()},0.25f);std::vector<float> output(model.output_features());
  const auto run=[&]{model.predict_proba_into({input.data(),input.size()},1,output,workspace);};
  for(int i=0;i<100;++i)run();
  std::vector<double> times;times.reserve(2000);
  for(int i=0;i<2000;++i) {
   const auto start=std::chrono::steady_clock::now();run();
   times.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());
  }
  std::sort(times.begin(),times.end());
  std::cout<<"{\"backend\":\""<<inference_backend()<<"\",\"scope\":\"native prepared API including clock overhead\","
   <<"\"batch\":1,\"repetitions\":2000,\"scratch_bytes\":"<<workspace.scratch_bytes()
   <<",\"p50_us\":"<<times[1000]<<",\"p95_us\":"<<times[1900]<<",\"p99_us\":"<<times[1980]<<"}\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
