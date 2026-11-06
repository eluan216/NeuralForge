#include "neuralforge/adaptation.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>
namespace neuralforge {
namespace {
double accuracy(const MLPModel& model,const LabeledBatch& data,std::size_t minimum) {
 if(data.inputs.ndim()!=2||data.inputs.shape()[0]!=data.labels.size()||data.labels.size()<minimum)
  throw std::invalid_argument("Evaluation needs enough labelled rows with matching shape");
 for(auto label:data.labels)if(label>=model.output_features())throw std::invalid_argument("Evaluation label outside class range");
 const auto predictions=model.predict(data.inputs);std::size_t correct=0;
 for(std::size_t i=0;i<predictions.size();++i)correct+=predictions[i]==data.labels[i];
 return static_cast<double>(correct)/static_cast<double>(predictions.size());
}
}
AdaptiveDeployment::AdaptiveDeployment(MLPModel active):active_(std::move(active)) {(void)active_.prepare();}
GateReport AdaptiveDeployment::consider(const MLPModel& candidate,const LabeledBatch& recent,
                                        const LabeledBatch& reference,GatePolicy policy,bool approved) {
 if(!std::isfinite(policy.minimum_gain)||!std::isfinite(policy.maximum_retention_drop)||
    policy.minimum_gain<0||policy.minimum_gain>1||policy.maximum_retention_drop<0||policy.maximum_retention_drop>1||policy.minimum_samples==0)
  throw std::invalid_argument("Invalid promotion policy");
 if(candidate.input_features()!=active_.input_features()||candidate.output_features()!=active_.output_features())
  throw std::invalid_argument("Candidate input and class contracts must match active model");
 GateReport report;
 report.active_recent=accuracy(active_,recent,policy.minimum_samples);
 report.candidate_recent=accuracy(candidate,recent,policy.minimum_samples);
 report.active_reference=accuracy(active_,reference,policy.minimum_samples);
 report.candidate_reference=accuracy(candidate,reference,policy.minimum_samples);
 report.eligible=report.candidate_recent-report.active_recent>=policy.minimum_gain &&
  report.active_reference-report.candidate_reference<=policy.maximum_retention_drop;
 report.promoted=report.eligible&&approved;
 report.reason=!report.eligible?"Candidate failed recent-gain or retention criteria":
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
