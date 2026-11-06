#include "neuralforge/artifact.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
namespace neuralforge {
namespace {
constexpr std::size_t limit=16*1024*1024;
void finite(float x) {if(!std::isfinite(x))throw std::invalid_argument("Non-finite preprocessing value");}
void check_buffers(std::span<const float> x,std::span<float> y,std::size_t batch,std::size_t ni,std::size_t no) {
 if(batch>std::numeric_limits<std::size_t>::max()/ni||batch>std::numeric_limits<std::size_t>::max()/no||
    x.size()!=batch*ni||y.size()!=batch*no)throw std::invalid_argument("Artifact inference buffer size mismatch");
 if(x.empty()||y.empty())return;
 const auto a=reinterpret_cast<std::uintptr_t>(x.data()),b=reinterpret_cast<std::uintptr_t>(y.data());
 if(x.size()>std::numeric_limits<std::uintptr_t>::max()/sizeof(float)||y.size()>std::numeric_limits<std::uintptr_t>::max()/sizeof(float))
  throw std::invalid_argument("Artifact byte count overflow");
 const auto ax=x.size()*sizeof(float),by=y.size()*sizeof(float);
 if(a>std::numeric_limits<std::uintptr_t>::max()-ax||b>std::numeric_limits<std::uintptr_t>::max()-by||(a<b+by&&b<a+ax))
  throw std::invalid_argument("Artifact inference buffers overlap");
}
}
Standardization Standardization::fit(const Tensor& x) {
 if(x.ndim()!=2||x.shape()[0]==0||x.shape()[1]==0)throw std::invalid_argument("Standardization needs nonempty training rows");
 const auto rows=x.shape()[0],cols=x.shape()[1];Standardization s;
 s.mean.resize(cols);s.scale.resize(cols);
 for(std::size_t j=0;j<cols;++j) {
  double mean=0,m2=0;
  for(std::size_t i=0;i<rows;++i) {
   const double value=x.at(i,j);if(!std::isfinite(value))throw std::invalid_argument("Non-finite training feature");
   const auto delta=value-mean;mean+=delta/static_cast<double>(i+1);m2+=delta*(value-mean);
  }
  const double sd=std::sqrt(std::max(0.0,m2/static_cast<double>(rows)));
  s.mean[j]=static_cast<float>(mean);s.scale[j]=sd<1e-6?1.0f:static_cast<float>(sd);
  finite(s.mean[j]);finite(s.scale[j]);if(s.scale[j]<=0)throw std::invalid_argument("Invalid fitted scale");
 }
 return s;
}
Tensor Standardization::transform(const Tensor& x) const {
 if(x.ndim()!=2||x.shape()[1]!=mean.size()||mean.size()!=scale.size())throw std::invalid_argument("Standardization shape mismatch");
 for(std::size_t j=0;j<scale.size();++j){finite(mean[j]);finite(scale[j]);if(scale[j]<=0)throw std::invalid_argument("Invalid scale");}
 Tensor out(x.shape());
 for(std::size_t i=0;i<x.size();++i) {
  finite(x.data()[i]);const auto j=i%mean.size();
  out.data()[i]=static_cast<float>((static_cast<double>(x.data()[i])-mean[j])/scale[j]);finite(out.data()[i]);
 }
 return out;
}
InferenceArtifact::InferenceArtifact(MLPModel model,FeatureSchema schema,Standardization preprocessing)
 :model_(std::move(model)),schema_(std::move(schema)),preprocessing_(std::move(preprocessing)) {
 (void)model_.prepare();validate_schema(schema_,input_features(),output_features());
 if(preprocessing_.mean.empty()&&preprocessing_.scale.empty()) {
  preprocessing_.mean.assign(input_features(),0);preprocessing_.scale.assign(input_features(),1);
 }
 if(preprocessing_.mean.size()!=input_features()||preprocessing_.scale.size()!=input_features())
  throw std::invalid_argument("Preprocessing dimensions do not match model");
 for(std::size_t j=0;j<input_features();++j) {
  finite(preprocessing_.mean[j]);finite(preprocessing_.scale[j]);
  if(preprocessing_.scale[j]<=0)throw std::invalid_argument("Preprocessing scales must be positive");
  identity_=identity_&&preprocessing_.mean[j]==0&&preprocessing_.scale[j]==1;
 }
}
ArtifactWorkspace InferenceArtifact::prepare() const {
 ArtifactWorkspace w{model_.prepare(),{}};if(!identity_)w.normalized.resize(input_features());return w;
}
void InferenceArtifact::predict_proba_into(std::span<const float> raw,std::size_t batch,std::span<float> out,ArtifactWorkspace& w) const {
 check_buffers(raw,out,batch,input_features(),output_features());
 if(w.normalized.size()!=(identity_?0:input_features()))throw std::invalid_argument("Artifact workspace preprocessing mismatch");
 if(identity_){model_.predict_proba_into(raw,batch,out,w.model);return;}
 for(std::size_t row=0;row<batch;++row) {
  for(std::size_t j=0;j<input_features();++j) {
   finite(raw[row*input_features()+j]);
   w.normalized[j]=static_cast<float>((static_cast<double>(raw[row*input_features()+j])-preprocessing_.mean[j])/preprocessing_.scale[j]);
   finite(w.normalized[j]);
  }
  model_.predict_proba_into(w.normalized,1,out.subspan(row*output_features(),output_features()),w.model);
 }
}
Tensor InferenceArtifact::predict_proba(const Tensor& x) const {
 if(x.ndim()!=2||x.shape()[1]!=input_features())throw std::invalid_argument("Artifact input shape mismatch");
 Tensor y({x.shape()[0],output_features()});auto w=prepare();
 predict_proba_into({x.data(),x.size()},x.shape()[0],{y.data(),y.size()},w);return y;
}
std::vector<std::size_t> InferenceArtifact::predict(const Tensor& x) const {
 const auto p=predict_proba(x);std::vector<std::size_t> labels(p.shape()[0]);
 for(std::size_t i=0;i<labels.size();++i)for(std::size_t j=1;j<output_features();++j)
  if(p.at(i,j)>p.at(i,labels[i]))labels[i]=j;
 return labels;
}
std::uint64_t InferenceArtifact::fingerprint() const noexcept {
 auto hash=model_.fingerprint();
 const auto feed=[&](std::uint64_t x,int bytes){for(int i=0;i<bytes;++i){hash^=(x>>(8*i))&255;hash*=1099511628211ULL;}};
 // Include semantic names and preprocessing in the deployment identity.
 for(const auto* names:{&schema_.features,&schema_.classes})for(const auto& name:*names) {
  feed(name.size(),8);for(unsigned char c:name)feed(c,1);
 }
 for(const auto* values:{&preprocessing_.mean,&preprocessing_.scale})for(float value:*values)feed(std::bit_cast<std::uint32_t>(value),4);
 return hash;
}
void InferenceArtifact::save(const std::filesystem::path& path) const {
 if(input_features()>4096||output_features()>4096||parameter_bytes()/sizeof(float)>262144)
  throw std::invalid_argument("Bundle dimensions or parameter budget exceeded");
 static std::atomic<unsigned long long> counter{0};
 const auto temporary=std::filesystem::path(path.string()+".tmp-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+std::to_string(counter++));
 try {
  std::ofstream out(temporary,std::ios::binary);out.imbue(std::locale::classic());
  if(!out)throw std::runtime_error("Cannot create model bundle");
  out<<std::setprecision(std::numeric_limits<float>::max_digits10);
  out<<"NEURALFORGE_BUNDLE 1\n"<<input_features()<<' '<<output_features()<<"\nstandardize\n";
  for(const auto* names:{&schema_.features,&schema_.classes}) {
   for(const auto& name:*names)out<<std::quoted(name)<<' ';out<<'\n';
  }
  for(const auto* values:{&preprocessing_.mean,&preprocessing_.scale}) {
   for(float value:*values)out<<value<<' ';out<<'\n';
  }
  out<<"NEURALFORGE_MLP 1\n"<<input_features()<<' '<<model_.hidden_features()<<' '<<output_features()<<'\n';
  for(const auto& tensor:model_.parameters()){for(std::size_t i=0;i<tensor.size();++i)out<<tensor.at(i)<<' ';out<<'\n';}
  out.close();if(!out)throw std::runtime_error("Bundle write failed");std::filesystem::rename(temporary,path);
 }catch(...){std::error_code ec;std::filesystem::remove(temporary,ec);throw;}
}
InferenceArtifact InferenceArtifact::load(const std::filesystem::path& path) {
 std::ifstream source(path,std::ios::binary);if(!source)throw std::runtime_error("Cannot open model artifact");
 return read(source);
}
InferenceArtifact InferenceArtifact::read(std::istream& source) {
 std::string contents;char chunk[8192];
 while(source) {
  source.read(chunk,sizeof(chunk));const auto count=static_cast<std::size_t>(source.gcount());
  if(count>limit-contents.size())throw std::runtime_error("Artifact exceeds 16 MiB");
  contents.append(chunk,count);
 }
 if(!source.eof())throw std::runtime_error("Artifact read failed");
 std::istringstream in(contents);in.imbue(std::locale::classic());
 std::string magic;in>>magic;
 if(magic=="NEURALFORGE_MLP"){std::istringstream legacy(contents);legacy.imbue(std::locale::classic());return InferenceArtifact(MLPModel::read(legacy));}
 int version=0;long long inputs=0,outputs=0;std::string kind;
 if(magic!="NEURALFORGE_BUNDLE"||!(in>>version>>inputs>>outputs>>kind)||version!=1||
    inputs<1||outputs<1||inputs>4096||outputs>4096||kind!="standardize")throw std::runtime_error("Invalid model bundle header");
 FeatureSchema schema;schema.features.resize(static_cast<std::size_t>(inputs));schema.classes.resize(static_cast<std::size_t>(outputs));
 for(auto* names:{&schema.features,&schema.classes})for(auto& name:*names)if(!(in>>std::quoted(name)))throw std::runtime_error("Invalid bundle schema");
 Standardization s;s.mean.resize(static_cast<std::size_t>(inputs));s.scale.resize(static_cast<std::size_t>(inputs));
 for(auto* values:{&s.mean,&s.scale})for(auto& value:*values)if(!(in>>value))throw std::runtime_error("Invalid bundle preprocessing");
 auto model=MLPModel::read(in);
 if(model.input_features()!=static_cast<std::size_t>(inputs)||model.output_features()!=static_cast<std::size_t>(outputs)||model.parameter_bytes()/sizeof(float)>262144)
  throw std::runtime_error("Bundle and model contracts disagree");
 return InferenceArtifact(std::move(model),std::move(schema),std::move(s));
}
}
