#include <neuralforge/c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(int argc,char** argv) {
 if(argc!=2){fprintf(stderr,"Usage: %s model.nfmodel\n",argv[0]);return 2;}
 nf_model* model=NULL;nf_workspace* workspace=NULL;float* x=NULL;float* y=NULL;int result=1;
 if(nf_model_load(argv[1],&model)!=NF_OK)goto done;
 if(nf_workspace_create(model,&workspace)!=NF_OK)goto done;
 const size_t ni=nf_model_inputs(model),no=nf_model_outputs(model);
 x=(float*)calloc(ni,sizeof(float));y=(float*)calloc(no,sizeof(float));
 if(!x||!y)goto done;
 if(nf_predict_proba(model,workspace,x,ni,1,y,no)!=NF_OK)goto done;
 double sum=0;for(size_t j=0;j<no;++j){if(!isfinite(y[j]))goto done;sum+=y[j];}
 if(fabs(sum-1.0)>1e-5)goto done;
 printf("abi=%u backend=%s scratch_bytes=%zu classes=%zu\n",nf_abi_version(),nf_backend(),nf_workspace_bytes(workspace),no);
 result=0;
done:
 if(result)fprintf(stderr,"Inference failed: %s\n",nf_last_error());
 free(x);free(y);nf_workspace_destroy(workspace);nf_model_destroy(model);return result;
}
