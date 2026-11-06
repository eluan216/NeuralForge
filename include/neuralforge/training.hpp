#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include "neuralforge/artifact.hpp"
#include "neuralforge/optim.hpp"
namespace neuralforge {
class TrainableMLP {
 std::vector<Variable> parameters_;
 Adam optimizer_;
 FeatureSchema schema_;
 Standardization preprocessing_;
public:
 explicit TrainableMLP(const MLPModel& initial,float learning_rate=0.01f,FeatureSchema schema={},Standardization preprocessing={});
 TrainableMLP(const TrainableMLP&)=delete;
 TrainableMLP& operator=(const TrainableMLP&)=delete;
 TrainableMLP(TrainableMLP&&) noexcept=default;
 TrainableMLP& operator=(TrainableMLP&&) noexcept=default;
 float train_batch(const Tensor& inputs,const std::vector<std::size_t>& labels);
 [[nodiscard]] MLPModel snapshot() const;
 [[nodiscard]] InferenceArtifact artifact() const {return InferenceArtifact(snapshot(),schema_,preprocessing_);}
 [[nodiscard]] const Standardization& preprocessing() const noexcept {return preprocessing_;}
 [[nodiscard]] std::size_t steps() const noexcept {return optimizer_.timestep();}
 [[nodiscard]] const FeatureSchema& schema() const noexcept {return schema_;}
 void save_checkpoint(const std::filesystem::path& path) const;
 static TrainableMLP load_checkpoint(const std::filesystem::path& path);
};
}
