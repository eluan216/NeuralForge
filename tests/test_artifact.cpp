#include <gtest/gtest.h>
#include "edge_helpers.hpp"
#include "neuralforge/artifact.hpp"
#include "neuralforge/training.hpp"
#include "neuralforge/c_api.h"
#include <filesystem>
#include <fstream>
#include <sstream>
using namespace neuralforge;
TEST(Artifact, RawInputMatchesExplicitTrainingOnlyStandardization) {
 Tensor training{{1,10},{3,14}};const auto s=Standardization::fit(training);
 EXPECT_FLOAT_EQ(s.mean[0],2);EXPECT_FLOAT_EQ(s.mean[1],12);
 EXPECT_FLOAT_EQ(s.scale[0],1);EXPECT_FLOAT_EQ(s.scale[1],2);
 const auto model=nf_test::model(2,9,2);InferenceArtifact artifact(model,{{"x","y"},{"a","b"}},s);
 Tensor raw{{2,13},{4,17}};const auto expected=model.predict_proba(s.transform(raw)),actual=artifact.predict_proba(raw);
 for(std::size_t i=0;i<actual.size();++i)EXPECT_NEAR(actual.at(i),expected.at(i),2e-6);
 EXPECT_EQ(artifact.prepare().scratch_bytes(),(2+9)*sizeof(float));
}
TEST(Artifact, BundlePreservesWeightsNamesPreprocessingAndCInterface) {
 InferenceArtifact a(nf_test::model(2,9,2),{{"x","y"},{"a","b"}},{{2,12},{1,2}});
 const auto path=std::filesystem::temp_directory_path()/"nf_bundle_roundtrip.nfbundle";
 a.save(path);auto b=InferenceArtifact::load(path);EXPECT_EQ(a.fingerprint(),b.fingerprint());
 nf_model* cmodel=nullptr;nf_workspace* w=nullptr;
 ASSERT_EQ(nf_model_load(path.string().c_str(),&cmodel),NF_OK);
 ASSERT_EQ(nf_workspace_create(cmodel,&w),NF_OK);
 EXPECT_STREQ(nf_model_feature_name(cmodel,1),"y");
 EXPECT_STREQ(nf_model_class_name(cmodel,0),"a");
 EXPECT_EQ(nf_model_feature_name(cmodel,2),nullptr);
 float x[]{2,13},y[2]{};ASSERT_EQ(nf_predict_proba(cmodel,w,x,2,1,y,2),NF_OK);
 const auto p=b.predict_proba(Tensor{{2,13}});
 EXPECT_NEAR(p.at(0),y[0],2e-6);
 nf_workspace_destroy(w);nf_model_destroy(cmodel);std::filesystem::remove(path);
}
TEST(Artifact, RejectsMalformedBundlesAndInvalidScales) {
 for(const auto* raw:{"NEURALFORGE_BUNDLE 99\n","NEURALFORGE_BUNDLE 1\n999999 1\nstandardize\n",
  "NEURALFORGE_BUNDLE 1\n2 2\nunsupported\n"}) {
  std::istringstream in(raw);EXPECT_ANY_THROW(InferenceArtifact::read(in));
 }
 EXPECT_THROW(InferenceArtifact(nf_test::model(2,9,2),{},{{0,0},{0,1}}),std::invalid_argument);
 EXPECT_THROW(InferenceArtifact(nf_test::model(2,9,2),{{"x","x"},{"a","b"}}),std::invalid_argument);
}
TEST(Artifact, RejectsSemanticSchemaMismatchBeforePromotion) {
 AdaptiveDeployment deployment(InferenceArtifact(nf_test::identity(),{{"x","y"},{"a","b"}}));
 const auto id=deployment.active().fingerprint();
 EXPECT_THROW(deployment.consider(InferenceArtifact(nf_test::identity(),{{"y","x"},{"a","b"}}),
  nf_test::cohort(),nf_test::cohort(),{},true),std::invalid_argument);
 EXPECT_THROW(deployment.consider(InferenceArtifact(nf_test::identity(),{{"x","y"},{"b","a"}}),
  nf_test::cohort(),nf_test::cohort(),{},true),std::invalid_argument);
 EXPECT_EQ(deployment.active().fingerprint(),id);
}
TEST(Artifact, CheckpointResumesOnRawInputsWithPreprocessing) {
 const auto data=nf_test::cohort();auto s=Standardization::fit(data.inputs);
 TrainableMLP a(nf_test::model(2,9,2),0.02f,{},s);a.train_batch(data.inputs,data.labels);
 const auto path=std::filesystem::temp_directory_path()/"nf_normalized_checkpoint";
 a.save_checkpoint(path);auto b=TrainableMLP::load_checkpoint(path);
 EXPECT_EQ(a.artifact().fingerprint(),b.artifact().fingerprint());
 EXPECT_FLOAT_EQ(a.train_batch(data.inputs,data.labels),b.train_batch(data.inputs,data.labels));
 EXPECT_EQ(a.artifact().fingerprint(),b.artifact().fingerprint());std::filesystem::remove(path);
}
TEST(Artifact, ReadsLegacyIdentityCheckpoint) {
 const auto path=std::filesystem::temp_directory_path()/"nf_legacy_v1_checkpoint";
 {std::ofstream out(path);out<<"NEURALFORGE_TRAINING 1\n1 1 2\n0.01 0.9 0.999 1e-8 0\nidentity\n\"x\"\n\"a\" \"b\"\n1 0 1 -1 0 0\n";
  for(int i=0;i<12;++i)out<<"0 ";}
 auto model=TrainableMLP::load_checkpoint(path);EXPECT_EQ(model.steps(),0u);
 EXPECT_FLOAT_EQ(model.preprocessing().scale[0],1);std::filesystem::remove(path);
}
