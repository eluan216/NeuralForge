#include "neuralforge/training.hpp"
#include "neuralforge/loss.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <stdexcept>
namespace neuralforge {
namespace {
void write_tensor(std::ostream& out,const Tensor& t) {
 for(std::size_t i=0;i<t.size();++i)out<<t.data()[i]<<' ';
 out<<'\n';
}
void read_tensor(std::istream& in,Tensor& t,bool variance=false) {
 for(std::size_t i=0;i<t.size();++i) {
  float value=0;
  if(!(in>>value)||!std::isfinite(value)||(variance&&value<0))throw std::runtime_error("Invalid checkpoint tensor");
  t.data()[i]=value;
 }
}
}
TrainableMLP::TrainableMLP(const MLPModel& initial,float rate,FeatureSchema schema,Standardization preprocessing):optimizer_(rate),schema_(std::move(schema)),preprocessing_(std::move(preprocessing)) {
 if(!initial.input_features()||!initial.hidden_features()||!initial.output_features())throw std::invalid_argument("Uninitialized training model");
 for(auto& t:initial.parameters())parameters_.emplace_back(std::move(t),true);
 InferenceArtifact checked(initial,schema_,preprocessing_);
 schema_=checked.schema();preprocessing_=checked.preprocessing();
}
float TrainableMLP::train_batch(const Tensor& inputs,const std::vector<std::size_t>& labels) {
 if(inputs.ndim()!=2||inputs.shape()[0]!=labels.size()||inputs.shape()[1]!=parameters_[0].data().shape()[0]||labels.empty())
  throw std::invalid_argument("Training batch shape or labels invalid");
 for(std::size_t i=0;i<inputs.size();++i)if(!std::isfinite(inputs.data()[i]))throw std::invalid_argument("Non-finite training input");
 optimizer_.zero_grad(parameters_);
 Variable x(preprocessing_.transform(inputs),false);
 auto hidden=relu(add(matmul(x,parameters_[0]),parameters_[1]));
 auto logits=add(matmul(hidden,parameters_[2]),parameters_[3]);
 auto loss=cross_entropy_loss(logits,labels);
 const float value=loss.data().at(0);
 if(!std::isfinite(value))throw std::runtime_error("Non-finite training loss");
 loss.backward();optimizer_.step(parameters_);return value;
}
MLPModel TrainableMLP::snapshot() const {
 return MLPModel::from_parameters({parameters_[0].data(),parameters_[1].data(),parameters_[2].data(),parameters_[3].data()});
}
void TrainableMLP::save_checkpoint(const std::filesystem::path& path) const {
 const auto model=snapshot();const auto p=model.parameters();const auto state=optimizer_.snapshot(parameters_);
 std::size_t elements=0;for(const auto& t:p)elements+=t.size();
 if(model.input_features()>4096||model.hidden_features()>4096||model.output_features()>4096||elements>262144)throw std::invalid_argument("Checkpoint supports at most 262144 parameters");
 static std::atomic<unsigned long long> sequence{0};
 const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
 const auto temporary=std::filesystem::path(path.string()+".tmp-"+std::to_string(stamp)+"-"+std::to_string(sequence++));
 try {
  std::ofstream out(temporary,std::ios::binary);out.imbue(std::locale::classic());
  if(!out)throw std::runtime_error("Cannot create checkpoint staging file");
  out<<std::setprecision(std::numeric_limits<float>::max_digits10);
  out<<"NEURALFORGE_TRAINING 2\n"<<model.input_features()<<' '<<model.hidden_features()<<' '<<model.output_features()<<'\n';
  out<<state.learning_rate<<' '<<state.beta1<<' '<<state.beta2<<' '<<state.epsilon<<' '<<state.timestep<<'\n';
  out<<"standardize\n";
  for(const auto& name:schema_.features) {out<<std::quoted(name)<<' ';}
  out<<'\n';
  for(const auto& name:schema_.classes) {out<<std::quoted(name)<<' ';}
  out<<'\n';
  for(const auto* values:{&preprocessing_.mean,&preprocessing_.scale}) {for(float v:*values)out<<v<<' ';out<<'\n';}
  for(const auto& t:p)write_tensor(out,t);
  for(const auto& t:state.first)write_tensor(out,t);
  for(const auto& t:state.second)write_tensor(out,t);
  out.close();if(!out)throw std::runtime_error("Checkpoint write failed");
  // Atomic visibility on supported POSIX filesystems; not a power-loss durability guarantee.
  std::filesystem::rename(temporary,path);
 } catch(...) {std::error_code ec;std::filesystem::remove(temporary,ec);throw;}
}
TrainableMLP TrainableMLP::load_checkpoint(const std::filesystem::path& path) {
 if(std::filesystem::file_size(path)>16*1024*1024)throw std::runtime_error("Checkpoint exceeds 16 MiB");
 std::ifstream in(path,std::ios::binary);in.imbue(std::locale::classic());
 std::string magic;int version=0;long long ni=0,nh=0,no=0;AdamSnapshot state;
 if(!(in>>magic>>version)||magic!="NEURALFORGE_TRAINING"||(version!=1&&version!=2)||!(in>>ni>>nh>>no)||
    ni<1||nh<1||no<1||ni>4096||nh>4096||no>4096||ni*nh+nh+nh*no+no>262144)
  throw std::runtime_error("Invalid checkpoint header or parameter budget");
 const auto a=static_cast<std::size_t>(ni),b=static_cast<std::size_t>(nh),c=static_cast<std::size_t>(no);
 std::string step;
 if(!(in>>state.learning_rate>>state.beta1>>state.beta2>>state.epsilon>>step)||
    step.empty()||step.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("Invalid checkpoint optimizer");
 std::size_t consumed=0;
 const auto tick=std::stoull(step,&consumed);
 if(consumed!=step.size()||tick>std::numeric_limits<std::size_t>::max())throw std::runtime_error("Invalid checkpoint timestep");
 state.timestep=static_cast<std::size_t>(tick);
 std::string preprocessing;if(!(in>>preprocessing)||(version==1?preprocessing!="identity":preprocessing!="standardize"))throw std::runtime_error("Unsupported preprocessing");
 FeatureSchema schema;schema.features.resize(a);schema.classes.resize(c);
 for(auto& s:schema.features)if(!(in>>std::quoted(s)))throw std::runtime_error("Missing feature schema");
 for(auto& s:schema.classes)if(!(in>>std::quoted(s)))throw std::runtime_error("Missing class schema");
 Standardization normalization;
 if(version==2) {
  normalization.mean.resize(a);normalization.scale.resize(a);
  for(auto* values:{&normalization.mean,&normalization.scale})for(auto& value:*values)
   if(!(in>>value))throw std::runtime_error("Missing checkpoint preprocessing");
 }
 std::array<Tensor,4> p{Tensor({a,b}),Tensor({b}),Tensor({b,c}),Tensor({c})};
 for(auto& t:p)read_tensor(in,t);
 for(const auto& t:p)state.first.emplace_back(t.shape());
 for(const auto& t:p)state.second.emplace_back(t.shape());
 for(auto& t:state.first)read_tensor(in,t);
 for(auto& t:state.second)read_tensor(in,t,true);
 std::string trailing;if(in>>trailing)throw std::runtime_error("Unexpected checkpoint trailing data");
 TrainableMLP result(MLPModel::from_parameters(p),state.learning_rate,std::move(schema),std::move(normalization));
 result.optimizer_.restore(state,result.parameters_);return result;
}
}
