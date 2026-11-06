#include "neuralforge/adaptation.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <utility>
namespace neuralforge {
namespace {
struct Metrics {double accuracy;std::vector<double> recall;};
Metrics measure(const MLPModel& model,const LabeledBatch& data,std::size_t minimum,bool require_coverage=false) {
 if(data.inputs.ndim()!=2||data.inputs.shape()[0]!=data.labels.size()||data.labels.size()<minimum)
  throw std::invalid_argument("Evaluation needs enough labelled rows with matching shape");
 for(auto label:data.labels)if(label>=model.output_features())throw std::invalid_argument("Evaluation label outside class range");
 Tensor probabilities({data.labels.size(),model.output_features()});
 auto workspace=model.prepare();
 model.predict_proba_into({data.inputs.data(),data.inputs.size()},data.labels.size(),{probabilities.data(),probabilities.size()},workspace);
 std::size_t correct=0;std::vector<std::size_t> counts(model.output_features()),hits(model.output_features());
 for(std::size_t i=0;i<data.labels.size();++i) {
  std::size_t best=0;
  for(std::size_t j=1;j<model.output_features();++j)if(probabilities.at(i,j)>probabilities.at(i,best))best=j;
  ++counts[data.labels[i]];
  if(best==data.labels[i]){++correct;++hits[data.labels[i]];}
 }
 Metrics metrics{static_cast<double>(correct)/data.labels.size(),{}};
 for(std::size_t j=0;j<counts.size();++j) {
  if(require_coverage&&!counts[j])throw std::invalid_argument("Reference holdout must include every class");
  metrics.recall.push_back(counts[j]?static_cast<double>(hits[j])/counts[j]:0);
 }
 return metrics;
}
}
AdaptiveDeployment::AdaptiveDeployment(MLPModel active):active_(std::move(active)) {(void)active_.prepare();}
GateReport AdaptiveDeployment::consider(const MLPModel& candidate,const LabeledBatch& recent,
                                        const LabeledBatch& reference,GatePolicy policy,bool approved) {
 if(!std::isfinite(policy.minimum_gain)||!std::isfinite(policy.maximum_retention_drop)||
    policy.minimum_gain<0||policy.minimum_gain>1||policy.maximum_retention_drop<0||policy.maximum_retention_drop>1||policy.minimum_samples==0||!std::isfinite(policy.maximum_class_recall_drop)||policy.maximum_class_recall_drop<0||
    policy.maximum_class_recall_drop>1||!policy.maximum_parameter_bytes||!policy.maximum_scratch_bytes)
  throw std::invalid_argument("Invalid promotion policy");
 if(candidate.input_features()!=active_.input_features()||candidate.output_features()!=active_.output_features())
  throw std::invalid_argument("Candidate input and class contracts must match active model");
 GateReport report;
 const auto ar=measure(active_,recent,policy.minimum_samples),cr=measure(candidate,recent,policy.minimum_samples);
 const auto af=measure(active_,reference,policy.minimum_samples,true),cf=measure(candidate,reference,policy.minimum_samples,true);
 report.active_recent=ar.accuracy;report.candidate_recent=cr.accuracy;
 report.active_reference=af.accuracy;report.candidate_reference=cf.accuracy;
 for(std::size_t j=0;j<af.recall.size();++j)report.worst_reference_recall_drop=std::max(report.worst_reference_recall_drop,af.recall[j]-cf.recall[j]);
 report.candidate_parameter_bytes=candidate.parameter_bytes();
 report.candidate_scratch_bytes=candidate.prepare().scratch_bytes();
 report.resources_ok=report.candidate_parameter_bytes<=policy.maximum_parameter_bytes&&report.candidate_scratch_bytes<=policy.maximum_scratch_bytes;
 report.eligible=report.candidate_recent-report.active_recent>=policy.minimum_gain &&
  report.active_reference-report.candidate_reference<=policy.maximum_retention_drop&&
  report.worst_reference_recall_drop<=policy.maximum_class_recall_drop&&report.resources_ok;
 report.promoted=report.eligible&&approved;
 report.reason=!report.eligible?"Candidate failed gain, overall/per-class retention, or resource criteria":
  approved?"Candidate passed cohorts and was explicitly approved":"Candidate passed cohorts; explicit approval required";
 const auto before=active_.fingerprint(),after=report.promoted?candidate.fingerprint():before;
 // Stage all allocating state before changing the deployment.
 auto events=audit_;events.push_back({events.size()+1,report.promoted?"promoted":report.eligible?"awaiting_approval":"rejected",before,after,report});
 if(report.promoted) {
  auto next=candidate;auto previous=active_;
  previous_=std::move(previous);active_=std::move(next);
 }
 audit_.swap(events);return report;
}
bool AdaptiveDeployment::rollback(bool approved) {
 if(!approved||!previous_)return false;
 auto events=audit_;events.push_back({events.size()+1,"rollback",active_.fingerprint(),previous_->fingerprint(),{}});
 active_=std::move(*previous_);previous_.reset();audit_.swap(events);return true;
}
ErrorDriftMonitor::ErrorDriftMonitor(std::size_t window,double threshold):window_(window),threshold_(threshold) {
 if(window<2||window>4096||!std::isfinite(threshold)||threshold<=0||threshold>1)
  throw std::invalid_argument("Invalid error-monitor window or threshold");
 errors_.resize(2*window);
}
DriftSignal ErrorDriftMonitor::observe(bool prediction_correct) {
 errors_[next_]=static_cast<unsigned char>(!prediction_correct);next_=(next_+1)%errors_.size();
 if(count_<errors_.size())++count_;
 if(count_<errors_.size())return {};
 std::size_t old_errors=0,new_errors=0;
 for(std::size_t i=0;i<window_;++i) {
  old_errors+=errors_[(next_+i)%errors_.size()];
  new_errors+=errors_[(next_+window_+i)%errors_.size()];
 }
 const double old_rate=static_cast<double>(old_errors)/window_,new_rate=static_cast<double>(new_errors)/window_;
 return {true,old_rate,new_rate,new_rate-old_rate>=threshold_};
}
}
