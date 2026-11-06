#include <gtest/gtest.h>
#include "edge_helpers.hpp"
#include "neuralforge/c_api.h"
#include "neuralforge/kernels.hpp"
#include <atomic>
#include <cmath>
#include <limits>
#include <thread>
using namespace neuralforge;
TEST(Prepared, MatchesReferenceAcrossVectorTailsAndBatches) {
 for(auto width:{1u,7u,8u,9u,15u,16u,17u,31u,32u,33u}) {
  auto model=nf_test::model(5,width,11);auto workspace=model.prepare();
  Tensor input({7,5});for(std::size_t i=0;i<input.size();++i)input.at(i)=std::sin(static_cast<float>(i));
  const auto expected=model.predict_proba(input);std::vector<float> actual(expected.size());
  model.predict_proba_into({input.data(),input.size()},7,actual,workspace);
  for(std::size_t i=0;i<actual.size();++i)EXPECT_NEAR(actual[i],expected.at(i),2e-6)<<width<<":"<<i;
  EXPECT_EQ(workspace.scratch_bytes(),width*sizeof(float));
 }
}
TEST(Prepared, RejectsInvalidBuffersWorkspaceAndNonFiniteInput) {
 auto model=nf_test::model();auto w=model.prepare();std::vector<float> x(2,1),y(3);
 EXPECT_THROW(model.predict_proba_into(x,2,y,w),std::invalid_argument);
 auto other=nf_test::model(3,9,3).prepare();
 EXPECT_THROW(model.predict_proba_into(x,1,y,other),std::invalid_argument);
 x[0]=std::numeric_limits<float>::infinity();
 EXPECT_THROW(model.predict_proba_into(x,1,y,w),std::invalid_argument);
 std::vector<float> storage(5,0);
 EXPECT_THROW(model.predict_proba_into({storage.data(),2},1,{storage.data(),3},w),std::invalid_argument);
 EXPECT_NO_THROW(model.predict_proba_into({},0,{},w));
 EXPECT_THROW(model.predict_proba_into({},std::numeric_limits<std::size_t>::max(),{},w),std::invalid_argument);
}
TEST(Prepared, RejectsOverflowInsteadOfReturningNaN) {
 auto p=nf_test::model().parameters();p[0].at(0)=std::numeric_limits<float>::max();
 auto model=MLPModel::from_parameters(p);auto w=model.prepare();std::vector<float> x(2,100),y(3);
 EXPECT_THROW(model.predict_proba_into(x,1,y,w),std::runtime_error);
}
TEST(Prepared, SharedModelSupportsIndependentConcurrentWorkspaces) {
 const auto model=nf_test::model();const auto id=model.fingerprint();std::atomic<bool> ok{true};
 std::vector<std::thread> threads;
 for(int n=0;n<4;++n)threads.emplace_back([&]{
  auto w=model.prepare();std::vector<float> x(2,1),y(3);
  for(int i=0;i<100;++i){model.predict_proba_into(x,1,y,w);if(std::abs(y[0]+y[1]+y[2]-1)>1e-5)ok=false;}
 });
 for(auto& t:threads)t.join();
 EXPECT_TRUE(ok);
 EXPECT_EQ(id,model.fingerprint());
}
TEST(CAbi, InvalidArgumentsNeverThrowAcrossBoundary) {
 nf_model* model=reinterpret_cast<nf_model*>(1);nf_workspace* w=reinterpret_cast<nf_workspace*>(1);
 EXPECT_EQ(nf_model_load(nullptr,&model),NF_INVALID_ARGUMENT);
 EXPECT_EQ(model,nullptr);
 EXPECT_EQ(nf_workspace_create(nullptr,&w),NF_INVALID_ARGUMENT);
 EXPECT_EQ(w,nullptr);
 EXPECT_EQ(nf_model_load("missing",nullptr),NF_INVALID_ARGUMENT);
 EXPECT_EQ(nf_predict_proba(nullptr,nullptr,nullptr,0,0,nullptr,0),NF_INVALID_ARGUMENT);
 EXPECT_NE(nf_last_error()[0],'\0');
 EXPECT_EQ(nf_abi_version(),1u);
 EXPECT_NO_THROW(nf_model_destroy(nullptr));
 EXPECT_NO_THROW(nf_workspace_destroy(nullptr));
}
TEST(ModelIdentity, ChangesWithParametersAndSurvivesCopy) {
 const auto a=nf_test::model();auto p=a.parameters();p[1].at(0)+=1;
 EXPECT_EQ(a.fingerprint(),MLPModel::from_parameters(a.parameters()).fingerprint());
 EXPECT_NE(a.fingerprint(),MLPModel::from_parameters(p).fingerprint());
}
