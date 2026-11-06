#include "neuralforge/c_api.h"
#include "neuralforge/model.hpp"
#include "neuralforge/kernels.hpp"
#include <cstdio>
#include <exception>
#include <new>
#include <span>
struct nf_model {neuralforge::MLPModel value;};
struct nf_workspace {neuralforge::InferenceWorkspace value;};
namespace {
thread_local char error_text[256]{};
nf_status fail(nf_status status,const char* text) noexcept {std::snprintf(error_text,sizeof(error_text),"%s",text);return status;}
void clear() noexcept {error_text[0]='\0';}
template<class F>nf_status guarded(F&& f) noexcept {
 clear();try{f();return NF_OK;}
 catch(const std::invalid_argument& e){return fail(NF_INVALID_ARGUMENT,e.what());}
 catch(const std::exception& e){return fail(NF_RUNTIME_ERROR,e.what());}
 catch(...){return fail(NF_RUNTIME_ERROR,"Unknown runtime error");}
}
}
extern "C" {
unsigned nf_abi_version(void){return 1;}
const char* nf_backend(void){return neuralforge::inference_backend();}
const char* nf_last_error(void){return error_text;}
nf_status nf_model_load(const char* path,nf_model** out){
 if(!out) {return fail(NF_INVALID_ARGUMENT,"Output handle is null");}
 *out=nullptr;
 if(!path)return fail(NF_INVALID_ARGUMENT,"Model path is null");
 return guarded([&]{*out=new nf_model{neuralforge::MLPModel::load(path)};});
}
void nf_model_destroy(nf_model* model){delete model;clear();}
size_t nf_model_inputs(const nf_model* model){return model?model->value.input_features():0;}
size_t nf_model_outputs(const nf_model* model){return model?model->value.output_features():0;}
size_t nf_model_parameter_bytes(const nf_model* model){return model?model->value.parameter_bytes():0;}
nf_status nf_workspace_create(const nf_model* model,nf_workspace** out){
 if(!out) {return fail(NF_INVALID_ARGUMENT,"Output handle is null");}
 *out=nullptr;
 if(!model)return fail(NF_INVALID_ARGUMENT,"Model is null");
 return guarded([&]{*out=new nf_workspace{model->value.prepare()};});
}
void nf_workspace_destroy(nf_workspace* workspace){delete workspace;clear();}
size_t nf_workspace_bytes(const nf_workspace* workspace){return workspace?workspace->value.scratch_bytes():0;}
nf_status nf_predict_proba(const nf_model* model,nf_workspace* w,const float* x,size_t nx,size_t batch,float* y,size_t ny){
 if(!model||!w||(!x&&nx)||(!y&&ny))return fail(NF_INVALID_ARGUMENT,"Invalid inference handle or buffer");
 return guarded([&]{model->value.predict_proba_into(std::span<const float>(x,nx),batch,std::span<float>(y,ny),w->value);});
}
}
