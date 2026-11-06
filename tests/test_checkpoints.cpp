#include <gtest/gtest.h>
#include "edge_helpers.hpp"
#include "neuralforge/training.hpp"
#include <filesystem>
#include <fstream>
#include <limits>
using namespace neuralforge;
TEST(Checkpoint, ResumeMatchesUninterruptedAdamExactly) {
 const auto data=nf_test::cohort();TrainableMLP original(nf_test::model(2,9,2),0.03f,{{"x","y"},{"a","b"}});
 for(int i=0;i<7;++i)original.train_batch(data.inputs,data.labels);
 const auto path=std::filesystem::temp_directory_path()/"nf_resume_test.nfcheckpoint";
 original.save_checkpoint(path);auto restored=TrainableMLP::load_checkpoint(path);
 EXPECT_EQ(restored.steps(),7u);
 EXPECT_EQ(restored.schema().features,(std::vector<std::string>{"x","y"}));
 EXPECT_EQ(restored.snapshot().fingerprint(),original.snapshot().fingerprint());
 for(int i=0;i<8;++i) {
  EXPECT_FLOAT_EQ(original.train_batch(data.inputs,data.labels),restored.train_batch(data.inputs,data.labels));
  EXPECT_EQ(restored.snapshot().fingerprint(),original.snapshot().fingerprint());
 }
 std::filesystem::remove(path);
}
TEST(Checkpoint, RejectsTruncationOversizedHeadersAndTrailingTokens) {
 const auto path=std::filesystem::temp_directory_path()/"nf_bad_checkpoint";
 for(const auto* contents:{"","NEURALFORGE_TRAINING 99\n","NEURALFORGE_TRAINING 1\n99999999 99999999 2\n"}) {
  {std::ofstream out(path);out<<contents;}
  EXPECT_ANY_THROW(TrainableMLP::load_checkpoint(path));
 }
 TrainableMLP model(nf_test::model());model.save_checkpoint(path);
 {std::ofstream out(path,std::ios::app);out<<"trailing";}
 EXPECT_ANY_THROW(TrainableMLP::load_checkpoint(path));std::filesystem::remove(path);
}
TEST(Checkpoint, FailureDoesNotAlterWeightsOrStep) {
 TrainableMLP model(nf_test::model(2,9,2));const auto before=model.snapshot().fingerprint();
 auto data=nf_test::cohort();data.inputs.at(0)=std::numeric_limits<float>::quiet_NaN();
 EXPECT_THROW(model.train_batch(data.inputs,data.labels),std::invalid_argument);
 EXPECT_EQ(model.steps(),0u);
 EXPECT_EQ(model.snapshot().fingerprint(),before);
 EXPECT_THROW(TrainableMLP(nf_test::model(),0.01f,{{"x","x"},{"a","b","c"}}),std::invalid_argument);
}
TEST(AdamTransaction, OverflowPreservesAllParametersMomentsAndTimestep) {
 Variable a(Tensor({1},1),true),b(Tensor({1},2),true);std::vector<Variable> p{a,b};Adam optimizer;
 optimizer.zero_grad(p);a.grad().at(0)=1;b.grad().at(0)=std::numeric_limits<float>::max();
 EXPECT_THROW(optimizer.step(p),std::runtime_error);
 EXPECT_EQ(optimizer.timestep(),0u);
 EXPECT_FLOAT_EQ(a.data().at(0),1);
 EXPECT_FLOAT_EQ(b.data().at(0),2);
 const auto saved=optimizer.snapshot(p);
 EXPECT_FLOAT_EQ(saved.first[0].at(0),0);
}
TEST(AdamTransaction, InvalidRestorePreservesState) {
 Variable a(Tensor({1},1),true);std::vector<Variable> p{a};Adam optimizer;
 optimizer.zero_grad(p);a.grad().at(0)=1;optimizer.step(p);
 auto saved=optimizer.snapshot(p);saved.second[0].at(0)=-1;
 EXPECT_THROW(optimizer.restore(saved,p),std::invalid_argument);
 EXPECT_EQ(optimizer.timestep(),1u);
 EXPECT_GT(optimizer.snapshot(p).second[0].at(0),0);
}
