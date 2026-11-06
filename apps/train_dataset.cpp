#include "neuralforge/training.hpp"
#include "neuralforge/adaptation.hpp"
#include "neuralforge/export_model.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
using namespace neuralforge;
namespace {
struct Data {LabeledBatch batch;std::size_t features,classes;};
Data read_data(const std::filesystem::path& path) {
 if(std::filesystem::file_size(path)>256*1024*1024)throw std::runtime_error("Dataset file exceeds 256 MiB");
 std::ifstream in(path);in.imbue(std::locale::classic());
 std::string magic;int version=0;long long rows=0,features=0,classes=0;
 if(!(in>>magic>>version>>rows>>features>>classes)||magic!="NF_DATASET"||version!=1||rows<1||rows>100000||
    features<1||features>4096||classes<2||classes>256||rows*features>12000000)throw std::runtime_error("Invalid labelled dataset header");
 Data data{{Tensor({static_cast<std::size_t>(rows),static_cast<std::size_t>(features)}),{}},
  static_cast<std::size_t>(features),static_cast<std::size_t>(classes)};
 std::vector<std::size_t> coverage(data.classes);
 for(std::size_t i=0;i<static_cast<std::size_t>(rows);++i) {
  for(std::size_t j=0;j<data.features;++j)
   if(!(in>>data.batch.inputs.at(i,j))||!std::isfinite(data.batch.inputs.at(i,j)))throw std::runtime_error("Invalid dataset feature");
  long long label=0;if(!(in>>label)||label<0||label>=classes)throw std::runtime_error("Invalid dataset label");
  data.batch.labels.push_back(static_cast<std::size_t>(label));++coverage[static_cast<std::size_t>(label)];
 }
 for(auto n:coverage)if(!n)throw std::runtime_error("Training and validation must each represent every class");
 in>>std::ws;if(in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Unexpected dataset trailing data");
 return data;
}
FeatureSchema read_schema(const std::filesystem::path& path,std::size_t ni,std::size_t nc) {
 FeatureSchema schema;
 if(path.empty()){validate_schema(schema,ni,nc);return schema;}
 if(std::filesystem::file_size(path)>2*1024*1024)throw std::runtime_error("Schema exceeds size limit");
 std::ifstream in(path);std::string magic;int version=0;std::size_t a=0,b=0;
 if(!(in>>magic>>version>>a>>b)||magic!="NEURALFORGE_SCHEMA"||version!=1||a!=ni||b!=nc)throw std::runtime_error("Schema header mismatch");
 schema.features.resize(a);schema.classes.resize(b);
 for(auto* names:{&schema.features,&schema.classes})for(auto& name:*names)
  if(!(in>>std::quoted(name)))throw std::runtime_error("Missing schema name");
 in>>std::ws;if(in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Unexpected schema trailing data");
 validate_schema(schema,ni,nc);return schema;
}
struct Score {double accuracy,macro_f1;};
Score score(const InferenceArtifact& artifact,const LabeledBatch& data) {
 const auto prediction=artifact.predict(data.inputs);const auto classes=artifact.output_features();
 std::vector<std::size_t> tp(classes),actual(classes),predicted(classes);std::size_t correct=0;
 for(std::size_t i=0;i<prediction.size();++i){++actual[data.labels[i]];++predicted[prediction[i]];if(prediction[i]==data.labels[i]){++correct;++tp[prediction[i]];}}
 double f1=0;for(std::size_t j=0;j<classes;++j)f1+=actual[j]+predicted[j]?2.0*tp[j]/(actual[j]+predicted[j]):0;
 return {static_cast<double>(correct)/prediction.size(),f1/classes};
}
std::size_t integer(const std::string& text,std::size_t low,std::size_t high) {
 if(text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("Expected a positive integer option");
 std::size_t pos=0;auto n=std::stoull(text,&pos);if(pos!=text.size()||n<low||n>high)throw std::invalid_argument("Integer option outside limits");
 return static_cast<std::size_t>(n);
}
}
int main(int argc,char** argv) {
 try {
  std::filesystem::path train_path,validation_path,schema_path,output;
  std::size_t epochs=40,hidden=32,batch_size=128,seed=7;float rate=0.003f;
  for(int i=1;i<argc;++i) {
   const std::string option=argv[i];
   if(i+1>=argc)throw std::invalid_argument("Every option needs a value");
   const std::string value=argv[++i];
   if(option=="--train")train_path=value;
   else if(option=="--validation")validation_path=value;
   else if(option=="--schema")schema_path=value;
   else if(option=="--output")output=value;
   else if(option=="--epochs")epochs=integer(value,1,500);
   else if(option=="--hidden")hidden=integer(value,1,512);
   else if(option=="--batch")batch_size=integer(value,1,4096);
   else if(option=="--seed")seed=integer(value,0,1000000);
   else if(option=="--learning-rate"){std::size_t pos=0;rate=std::stof(value,&pos);if(pos!=value.size()||!std::isfinite(rate)||rate<=0||rate>1)throw std::invalid_argument("Invalid learning rate");}
   else throw std::invalid_argument("Unknown dataset-training option");
  }
  if(train_path.empty()||validation_path.empty()||output.empty())throw std::invalid_argument("Required: --train data --validation data --output directory");
  const auto train=read_data(train_path),validation=read_data(validation_path);
  if(train.features!=validation.features||train.classes!=validation.classes)throw std::runtime_error("Training and validation dimensions differ");
  if(train.features*hidden+hidden+hidden*train.classes+train.classes>262144)throw std::runtime_error("Requested model exceeds artifact parameter budget");
  const auto schema=read_schema(schema_path,train.features,train.classes);
  const auto normalization=Standardization::fit(train.batch.inputs);
  std::mt19937 random(static_cast<unsigned>(seed));std::normal_distribution<float> init(0,std::sqrt(2.0f/static_cast<float>(train.features)));
  std::array<Tensor,4> p{Tensor({train.features,hidden}),Tensor({hidden},0),Tensor({hidden,train.classes}),Tensor({train.classes},0)};
  for(auto index:{0,2})for(std::size_t i=0;i<p[index].size();++i)p[index].at(i)=init(random);
  TrainableMLP learner(MLPModel::from_parameters(p),rate,schema,normalization);
  std::filesystem::create_directories(output);
  std::ofstream history(output/"history.csv");history<<"epoch,training_loss,validation_accuracy,validation_macro_f1\n";
  std::vector<std::size_t> order(train.batch.labels.size());std::iota(order.begin(),order.end(),0);
  double best=-1;std::size_t best_epoch=0;Score best_score{};
  for(std::size_t epoch=1;epoch<=epochs;++epoch) {
   std::shuffle(order.begin(),order.end(),random);double loss=0;
   for(std::size_t first=0;first<order.size();first+=batch_size) {
    const auto count=std::min(batch_size,order.size()-first);Tensor x({count,train.features});std::vector<std::size_t> labels;labels.reserve(count);
    for(std::size_t row=0;row<count;++row) {
     const auto index=order[first+row];std::copy_n(train.batch.inputs.data()+index*train.features,train.features,x.data()+row*train.features);labels.push_back(train.batch.labels[index]);
    }
    loss+=learner.train_batch(x,labels)*count;
   }
   const auto artifact=learner.artifact();const auto evaluated=score(artifact,validation.batch);
   history<<epoch<<','<<loss/order.size()<<','<<evaluated.accuracy<<','<<evaluated.macro_f1<<'\n';
   if(evaluated.macro_f1>best) {
    best=evaluated.macro_f1;best_epoch=epoch;best_score=evaluated;
    artifact.save(output/"model.nfbundle");learner.save_checkpoint(output/"best.nfcheckpoint");
    const auto weights=learner.snapshot().parameters();
    export_mlp_model(output/"weights.nfmodel",weights[0],weights[1],weights[2],weights[3]);
   }
  }
  history.close();if(!history)throw std::runtime_error("Cannot write training history");
  const auto selected=InferenceArtifact::load(output/"model.nfbundle");
  std::ofstream metadata(output/"training.json");metadata<<std::setprecision(9);
  metadata<<"{\n\"training_rows\":"<<train.batch.labels.size()<<",\"validation_rows\":"<<validation.batch.labels.size()
   <<",\"input_features\":"<<train.features<<",\"classes\":"<<train.classes<<",\"hidden\":"<<hidden
   <<",\"seed\":"<<seed<<",\"epochs\":"<<epochs<<",\"batch_size\":"<<batch_size<<",\"learning_rate\":"<<rate
   <<",\"selection\":\"validation macro-F1; earliest best epoch\",\"selected_epoch\":"<<best_epoch
   <<",\"validation_accuracy\":"<<best_score.accuracy<<",\"validation_macro_f1\":"<<best_score.macro_f1
   <<",\"artifact_id\":\""<<selected.fingerprint()<<"\",\"mean\":[";
  bool first=true;for(float v:selected.preprocessing().mean){if(!first)metadata<<',';first=false;metadata<<v;}
  metadata<<"],\"scale\":[";first=true;for(float v:selected.preprocessing().scale){if(!first)metadata<<',';first=false;metadata<<v;}
  metadata<<"],\"feature_names\":[";first=true;for(const auto& name:selected.schema().features){if(!first)metadata<<',';first=false;metadata<<std::quoted(name);}
  metadata<<"],\"class_names\":[";first=true;for(const auto& name:selected.schema().classes){if(!first)metadata<<',';first=false;metadata<<std::quoted(name);}
  metadata<<"]}\n";metadata.close();if(!metadata)throw std::runtime_error("Cannot write training metadata");
  std::cout<<"selected_epoch="<<best_epoch<<" validation_accuracy="<<best_score.accuracy<<" validation_macro_f1="<<best_score.macro_f1<<'\n';
  return 0;
 }catch(const std::exception& error){std::cerr<<"train_dataset: "<<error.what()<<'\n';return 1;}
}
