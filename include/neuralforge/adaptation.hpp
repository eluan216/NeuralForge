#pragma once
#include <optional>
#include <string>
#include <vector>
#include "neuralforge/model.hpp"
namespace neuralforge {
struct LabeledBatch {Tensor inputs;std::vector<std::size_t> labels;};
struct GatePolicy {double minimum_gain=0.02;double maximum_retention_drop=0.01;std::size_t minimum_samples=20;};
struct GateReport {
 double active_recent{},candidate_recent{},active_reference{},candidate_reference{};
 bool eligible=false,promoted=false;std::string reason;
};
struct AuditEvent {
 std::size_t sequence;std::string action;std::uint64_t before,after;GateReport report;
};
class AdaptiveDeployment {
 MLPModel active_;
 std::optional<MLPModel> previous_;
 std::vector<AuditEvent> audit_;
public:
 explicit AdaptiveDeployment(MLPModel active);
 [[nodiscard]] const MLPModel& active() const noexcept {return active_;}
 [[nodiscard]] const std::vector<AuditEvent>& audit() const noexcept {return audit_;}
 /// Caller must supply untouched, representative recent and reference holdouts.
 /// Promotion evaluates both cohorts internally and requires explicit approval.
 GateReport consider(const MLPModel& candidate,const LabeledBatch& recent,const LabeledBatch& reference,
                     GatePolicy policy={},bool approved=false);
 bool rollback(bool approved=false);
};
struct DriftSignal {bool ready=false;double reference_error{},recent_error{};bool alert=false;};
/// Bounded-memory, labelled-error monitor. A signal is not proof that retraining is appropriate.
class ErrorDriftMonitor {
 std::vector<unsigned char> errors_;
 std::size_t window_,next_=0,count_=0;double threshold_;
public:
 explicit ErrorDriftMonitor(std::size_t window=32,double threshold=0.20);
 DriftSignal observe(bool prediction_correct);
};
}
