#include "neuralforge/adaptation.hpp"
#include "neuralforge/training.hpp"
#include "neuralforge/export_model.hpp"
#include "neuralforge/kernels.hpp"
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace neuralforge;
namespace {
LabeledBatch sensor(std::size_t n,float offset,unsigned seed) {
 std::mt19937 rng(seed);std::normal_distribution<float> noise(0,0.15f);
 LabeledBatch b{Tensor({n*2,2}),{}};
 for(std::size_t c=0;c<2;++c)for(std::size_t i=0;i<n;++i) {
  const auto row=c*n+i;b.inputs.at(row,0)=(c?2.0f:-2.0f)+offset+noise(rng);
  b.inputs.at(row,1)=(c?1.0f:-1.0f)+noise(rng);b.labels.push_back(c);
 }
 return b;
}
LabeledBatch combine(const LabeledBatch& a,const LabeledBatch& b) {
 LabeledBatch out{Tensor({a.labels.size()+b.labels.size(),2}),a.labels};
 std::copy_n(a.inputs.data(),a.inputs.size(),out.inputs.data());
 std::copy_n(b.inputs.data(),b.inputs.size(),out.inputs.data()+a.inputs.size());
 out.labels.insert(out.labels.end(),b.labels.begin(),b.labels.end());return out;
}
}
int main(int argc,char** argv) {
 try {
  std::filesystem::path directory="artifacts/adaptation";
  for(int i=1;i<argc;++i) {
   if(std::string(argv[i])=="--output"&&i+1<argc)directory=argv[++i];
   else throw std::invalid_argument("Usage: neuralforge_adapt_demo [--output directory]");
  }
  std::filesystem::create_directories(directory);
  std::mt19937 rng(7);std::normal_distribution<float> init(0,0.1f);
  std::array<Tensor,4> p{Tensor({2,16}),Tensor({16},0),Tensor({16,2}),Tensor({2},0)};
  for(auto index:{0,2})for(std::size_t i=0;i<p[index].size();++i)p[index].at(i)=init(rng);
  TrainableMLP learner(MLPModel::from_parameters(p),0.03f,{{"sensor_x","sensor_y"},{"idle","running"}});
  const auto initial_train=sensor(80,0,11);
  for(int i=0;i<100;++i)learner.train_batch(initial_train.inputs,initial_train.labels);
  const auto baseline=learner.snapshot();
  learner.save_checkpoint(directory/"baseline.nfcheckpoint");
  auto candidate=TrainableMLP::load_checkpoint(directory/"baseline.nfcheckpoint");
  const auto shifted_train=sensor(80,6,21),reference_holdout=sensor(60,0,31),recent_holdout=sensor(60,6,41);
  const auto replay=combine(initial_train,shifted_train);
  for(int i=0;i<120;++i)candidate.train_batch(replay.inputs,replay.labels);
  candidate.save_checkpoint(directory/"candidate.nfcheckpoint");
  const auto model=candidate.snapshot();
  AdaptiveDeployment deployment(baseline);
  GatePolicy policy{0.05,0.02,30};
  const auto review=deployment.consider(model,recent_holdout,reference_holdout,policy,false);
  if(!review.eligible||review.promoted)throw std::runtime_error("Demo candidate did not pass review without promotion");
  const auto promotion=deployment.consider(model,recent_holdout,reference_holdout,policy,true);
  if(!promotion.promoted)throw std::runtime_error("Demo promotion failed");
  auto bad=p;for(auto& tensor:bad)for(std::size_t i=0;i<tensor.size();++i)tensor.at(i)=0;
  const auto rejection=deployment.consider(MLPModel::from_parameters(bad),recent_holdout,reference_holdout,policy,true);
  if(rejection.promoted||rejection.eligible)throw std::runtime_error("Bad candidate was not rejected");
  if(!deployment.rollback(true)||deployment.active().fingerprint()!=baseline.fingerprint())
   throw std::runtime_error("Rollback did not restore original weights");
  const auto weights=model.parameters();
  export_mlp_model(directory/"candidate.nfmodel",weights[0],weights[1],weights[2],weights[3]);
  ErrorDriftMonitor monitor(20,0.2);DriftSignal signal;
  const auto old_predictions=baseline.predict(reference_holdout.inputs),new_predictions=baseline.predict(recent_holdout.inputs);
  for(std::size_t i=0;i<20;++i)monitor.observe(old_predictions[i]==reference_holdout.labels[i]);
  for(std::size_t i=0;i<20;++i)signal=monitor.observe(new_predictions[i]==recent_holdout.labels[i]);
  std::ofstream out(directory/"report.json");
  out<<"{\n  \"simulation\":true,\n  \"backend\":\""<<inference_backend()<<"\",\n"
   <<"  \"training_samples_initial\":"<<initial_train.labels.size()<<",\n  \"training_samples_replay\":"<<replay.labels.size()<<",\n"
   <<"  \"recent_holdout_samples\":"<<recent_holdout.labels.size()<<",\n  \"reference_holdout_samples\":"<<reference_holdout.labels.size()<<",\n"
   <<"  \"baseline_recent_accuracy\":"<<promotion.active_recent<<",\n  \"candidate_recent_accuracy\":"<<promotion.candidate_recent<<",\n"
   <<"  \"baseline_reference_accuracy\":"<<promotion.active_reference<<",\n  \"candidate_reference_accuracy\":"<<promotion.candidate_reference<<",\n"
   <<"  \"labelled_error_alert\":"<<(signal.alert?"true":"false")<<",\n"
   <<"  \"explicit_approval_required\":true,\n  \"promotion_verified\":true,\n  \"bad_candidate_rejected\":true,\n  \"rollback_verified\":true,\n"
   <<"  \"events\":[";
  bool first=true;for(const auto& event:deployment.audit()) {
   if(!first)out<<',';first=false;
   out<<"{\"sequence\":"<<event.sequence<<",\"action\":\""<<event.action<<"\",\"before\":\""<<event.before<<"\",\"after\":\""<<event.after<<"\"}";
  }
  out<<"]\n}\n";out.close();if(!out)throw std::runtime_error("Cannot write demo report");
  std::ifstream report(directory/"report.json");std::cout<<report.rdbuf();return 0;
 }catch(const std::exception& e){std::cerr<<"adapt_demo: "<<e.what()<<'\n';return 1;}
}
