#ifndef NEURALFORGE_C_API_H
#define NEURALFORGE_C_API_H
#include <stddef.h>
#if defined(_WIN32) && defined(NF_SHARED)
 #if defined(NF_BUILDING)
  #define NF_API __declspec(dllexport)
 #else
  #define NF_API __declspec(dllimport)
 #endif
#else
 #define NF_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct nf_model nf_model;
typedef struct nf_workspace nf_workspace;
typedef enum nf_status {NF_OK=0,NF_INVALID_ARGUMENT=1,NF_RUNTIME_ERROR=2} nf_status;
NF_API unsigned nf_abi_version(void);
NF_API const char* nf_backend(void);
/* Error text is thread-local and valid until the next API call on this thread. */
NF_API const char* nf_last_error(void);
NF_API nf_status nf_model_load(const char* path,nf_model** out);
NF_API void nf_model_destroy(nf_model* model);
NF_API size_t nf_model_inputs(const nf_model* model);
NF_API size_t nf_model_outputs(const nf_model* model);
NF_API nf_status nf_workspace_create(const nf_model* model,nf_workspace** out);
NF_API void nf_workspace_destroy(nf_workspace* workspace);
NF_API size_t nf_workspace_bytes(const nf_workspace* workspace);
/* Counts are float elements, not bytes. Caller owns valid nonoverlapping buffers.
   Model is immutable; each concurrent call needs a separate workspace.
   On failure outputs may be partially written and must be discarded. */
NF_API nf_status nf_predict_proba(const nf_model* model,nf_workspace* workspace,
 const float* inputs,size_t input_count,size_t batch,float* outputs,size_t output_count);
#ifdef __cplusplus
}
#endif
#endif
