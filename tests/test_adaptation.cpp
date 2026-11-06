#include <gtest/gtest.h>
#include "edge_helpers.hpp"
#include <limits>
using namespace neuralforge;
TEST(Adaptation, RequiresApprovalAndSupportsOneLevelRollback) {
 AdaptiveDeployment deployment(nf_test::identity(true));const auto original=deployment.active().fingerprint();
 const auto candidate=nf_test::identity();const auto data=nf_test::cohort();
 const auto review=deployment.consider(candidate,data,data,{0.2,0,10});
 EXPECT_TRUE(review.eligible);
 EXPECT_FALSE(review.promoted);
 EXPECT_EQ(deployment.active().fingerprint(),original);
 const auto promotion=deployment.consider(candidate,data,data,{0.2,0,10},true);
 EXPECT_TRUE(promotion.promoted);
 EXPECT_EQ(deployment.active().fingerprint(),candidate.fingerprint());
 EXPECT_FALSE(deployment.rollback());
 EXPECT_TRUE(deployment.rollback(true));
 EXPECT_EQ(deployment.active().fingerprint(),original);
 EXPECT_FALSE(deployment.rollback(true));
 EXPECT_EQ(deployment.audit().size(),3u);
}
TEST(Adaptation, RejectsForgettingEvenWhenRecentAccuracyImproves) {
 AdaptiveDeployment deployment(nf_test::identity());const auto before=deployment.active().fingerprint();
 const auto report=deployment.consider(nf_test::identity(true),nf_test::cohort(true),nf_test::cohort(),{0.1,0.01,10},true);
 EXPECT_FALSE(report.eligible);
 EXPECT_FALSE(report.promoted);
 EXPECT_EQ(deployment.active().fingerprint(),before);
 EXPECT_EQ(deployment.audit()[0].action,"rejected");
}
TEST(Adaptation, InvalidInputsLeaveDeploymentUnchanged) {
 AdaptiveDeployment deployment(nf_test::identity());const auto before=deployment.active().fingerprint();
 EXPECT_THROW(deployment.consider(nf_test::identity(),nf_test::cohort(false,2),nf_test::cohort()),std::invalid_argument);
 EXPECT_THROW(deployment.consider(nf_test::model(3),nf_test::cohort(),nf_test::cohort()),std::invalid_argument);
 EXPECT_THROW(deployment.consider(nf_test::identity(),nf_test::cohort(),nf_test::cohort(),{std::numeric_limits<double>::quiet_NaN(),0,1}),std::invalid_argument);
 EXPECT_EQ(deployment.active().fingerprint(),before);
 EXPECT_TRUE(deployment.audit().empty());
}
TEST(DriftMonitor, RequiresLabelsAndTwoWindowsAndDoesNotActOnModels) {
 ErrorDriftMonitor monitor(4,0.5);DriftSignal signal;
 for(int i=0;i<4;++i)EXPECT_FALSE(monitor.observe(true).ready);
 for(int i=0;i<3;++i)EXPECT_FALSE(monitor.observe(false).ready);
 signal=monitor.observe(false);
 EXPECT_TRUE(signal.ready);
 EXPECT_TRUE(signal.alert);
 EXPECT_DOUBLE_EQ(signal.reference_error,0);
 EXPECT_DOUBLE_EQ(signal.recent_error,1);
 for(int i=0;i<12;++i)signal=monitor.observe(true);
 EXPECT_TRUE(signal.ready);
 EXPECT_FALSE(signal.alert);
 EXPECT_DOUBLE_EQ(signal.recent_error,0);
 EXPECT_THROW(ErrorDriftMonitor(0),std::invalid_argument);
 EXPECT_THROW(ErrorDriftMonitor(4,0),std::invalid_argument);
}

TEST(Adaptation, EnforcesDeploymentMemoryBudgetAndClassCoverage) {
 AdaptiveDeployment deployment(nf_test::identity(true));GatePolicy policy{0.1,0.0,10};
 policy.maximum_scratch_bytes=1;
 const auto report=deployment.consider(nf_test::identity(),nf_test::cohort(),nf_test::cohort(),policy,true);
 EXPECT_FALSE(report.resources_ok);
 EXPECT_FALSE(report.promoted);
 auto missing=nf_test::cohort();for(auto& label:missing.labels)label=0;
 EXPECT_THROW(deployment.consider(nf_test::identity(),nf_test::cohort(),missing),std::invalid_argument);
}

TEST(Adaptation, ProtectsMinorityRecallWhenAggregateAccuracyBarelyDrops) {
 AdaptiveDeployment deployment(nf_test::identity());
 auto p=nf_test::identity().parameters();
 for(auto& t:p)for(std::size_t i=0;i<t.size();++i)t.at(i)=0;
 p[3].at(0)=1;
 const auto candidate=MLPModel::from_parameters(p);
 auto recent=nf_test::cohort();for(auto& label:recent.labels)label=0;
 LabeledBatch reference{Tensor({100,2}),std::vector<std::size_t>(100,0)};
 for(std::size_t i=0;i<100;++i){reference.inputs.at(i,0)=2;reference.inputs.at(i,1)=1;}
 reference.inputs.at(99,0)=1;reference.inputs.at(99,1)=2;reference.labels[99]=1;
 GatePolicy policy{0.1,0.02,20};
 const auto report=deployment.consider(candidate,recent,reference,policy,true);
 EXPECT_NEAR(report.active_reference-report.candidate_reference,0.01,1e-8);
 EXPECT_DOUBLE_EQ(report.worst_reference_recall_drop,1);
 EXPECT_FALSE(report.eligible);
 policy.maximum_class_recall_drop=1;
 EXPECT_TRUE(deployment.consider(candidate,recent,reference,policy).eligible);
}
