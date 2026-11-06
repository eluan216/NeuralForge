#include "neuralforge/model.hpp"
#include "neuralforge/artifact.hpp"
#include <filesystem>
#include "neuralforge/c_api.h"
#include "neuralforge/kernels.hpp"
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>
#include <vector>
#if defined(_WIN32)
#include <malloc.h>
#endif
namespace {std::atomic<bool> track{false};std::atomic<std::size_t> allocations{0};}
void* operator new(std::size_t n) {
 if(track.load())++allocations;
 if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();
}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}
void* operator new(std::size_t n,std::align_val_t a) {
 if(track.load())++allocations;void* p=nullptr;
#if defined(_WIN32)
 p=_aligned_malloc(n?n:1,static_cast<std::size_t>(a));
#else
 if(posix_memalign(&p,static_cast<std::size_t>(a),n?n:1)!=0)p=nullptr;
#endif
 if(!p)throw std::bad_alloc();return p;
}
void* operator new[](std::size_t n,std::align_val_t a){return ::operator new(n,a);}
void operator delete(void* p,std::align_val_t) noexcept {
#if defined(_WIN32)
 _aligned_free(p);
#else
 std::free(p);
#endif
}
void operator delete[](void* p,std::align_val_t a) noexcept {::operator delete(p,a);}
void operator delete(void* p,std::size_t,std::align_val_t a) noexcept {::operator delete(p,a);}
void operator delete[](void* p,std::size_t,std::align_val_t a) noexcept {::operator delete(p,a);}
int main(int argc,char** argv) {
 if(argc!=2)return 2;
 try {
  const auto model=neuralforge::MLPModel::load(argv[1]);auto workspace=model.prepare();
  std::vector<float> input(4*model.input_features(),0.25f),output(4*model.output_features());
  (void)neuralforge::inference_backend();
  nf_model* cmodel=nullptr;nf_workspace* cworkspace=nullptr;
  if(nf_model_load(argv[1],&cmodel)!=NF_OK||nf_workspace_create(cmodel,&cworkspace)!=NF_OK)return 3;
  neuralforge::Standardization normalization;
  normalization.mean.assign(model.input_features(),1);normalization.scale.assign(model.input_features(),2);
  neuralforge::InferenceArtifact artifact(model,{},normalization);auto artifact_workspace=artifact.prepare();
  const auto bundle_path=std::filesystem::temp_directory_path()/"nf_allocations.nfbundle";artifact.save(bundle_path);
  nf_model* bundled=nullptr;nf_workspace* bundled_workspace=nullptr;
  if(nf_model_load(bundle_path.string().c_str(),&bundled)!=NF_OK||nf_workspace_create(bundled,&bundled_workspace)!=NF_OK)return 5;
  track=true;
  for(int i=0;i<1000;++i) {
   model.predict_proba_into(input,4,output,workspace);
   artifact.predict_proba_into(input,4,output,artifact_workspace);
   if(nf_predict_proba(bundled,bundled_workspace,input.data(),input.size(),4,output.data(),output.size())!=NF_OK){track=false;return 6;}
   if(nf_predict_proba(cmodel,cworkspace,input.data(),input.size(),4,output.data(),output.size())!=NF_OK){track=false;return 4;}
  }
  track=false;nf_workspace_destroy(bundled_workspace);nf_model_destroy(bundled);std::filesystem::remove(bundle_path);
  nf_workspace_destroy(cworkspace);nf_model_destroy(cmodel);
  std::cout<<"prepared_cpp_and_c_calls=4000 cxx_heap_allocations="<<allocations.load()<<'\n';
  return allocations.load()==0?0:1;
 }catch(const std::exception& e){track=false;std::cerr<<e.what()<<'\n';return 1;}
}
