#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include "neuralforge/model.hpp"
#include "neuralforge/optim.hpp"
namespace neuralforge {
struct FeatureSchema {
 std::vector<std::string> features,classes;
 /// Only identity preprocessing is supported by this checkpoint version.
};
class TrainableMLP {
 std::vector<Variable> parameters_;
 Adam optimizer_;
 FeatureSchema schema_;
public:
 explicit TrainableMLP(const MLPModel& initial,float learning_rate=0.01f,FeatureSchema schema={});
 TrainableMLP(const TrainableMLP&)=delete;
 TrainableMLP& operator=(const TrainableMLP&)=delete;
 TrainableMLP(TrainableMLP&&) noexcept=default;
 TrainableMLP& operator=(TrainableMLP&&) noexcept=default;
 float train_batch(const Tensor& inputs,const std::vector<std::size_t>& labels);
 [[nodiscard]] MLPModel snapshot() const;
 [[nodiscard]] std::size_t steps() const noexcept {return optimizer_.timestep();}
 [[nodiscard]] const FeatureSchema& schema() const noexcept {return schema_;}
 void save_checkpoint(const std::filesystem::path& path) const;
 static TrainableMLP load_checkpoint(const std::filesystem::path& path);
};
}
