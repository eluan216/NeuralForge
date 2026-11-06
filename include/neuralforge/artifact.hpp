#pragma once
#include <filesystem>
#include <iosfwd>
#include "neuralforge/model.hpp"
#include "neuralforge/schema.hpp"
namespace neuralforge {
struct Standardization {
 std::vector<float> mean,scale;
 static Standardization fit(const Tensor& training);
 [[nodiscard]] Tensor transform(const Tensor& input) const;
};
struct ArtifactWorkspace {
 InferenceWorkspace model;
 std::vector<float> normalized;
 [[nodiscard]] std::size_t scratch_bytes() const noexcept {return model.scratch_bytes()+normalized.size()*sizeof(float);}
};
/// A deployable classifier: ordered schema + affine preprocessing + inference weights.
class InferenceArtifact {
 MLPModel model_;
 FeatureSchema schema_;
 Standardization preprocessing_;
 bool identity_=true;
public:
 explicit InferenceArtifact(MLPModel model,FeatureSchema schema={},Standardization preprocessing={});
 [[nodiscard]] const MLPModel& model() const noexcept {return model_;}
 [[nodiscard]] const FeatureSchema& schema() const noexcept {return schema_;}
 [[nodiscard]] const Standardization& preprocessing() const noexcept {return preprocessing_;}
 [[nodiscard]] std::size_t input_features() const noexcept {return model_.input_features();}
 [[nodiscard]] std::size_t output_features() const noexcept {return model_.output_features();}
 [[nodiscard]] std::size_t parameter_bytes() const noexcept {return model_.parameter_bytes();}
 [[nodiscard]] std::size_t preprocessing_bytes() const noexcept {return 2*input_features()*sizeof(float);}
 [[nodiscard]] ArtifactWorkspace prepare() const;
 void predict_proba_into(std::span<const float> raw,std::size_t batch,std::span<float> outputs,ArtifactWorkspace& workspace) const;
 [[nodiscard]] Tensor predict_proba(const Tensor& raw) const;
 [[nodiscard]] std::vector<std::size_t> predict(const Tensor& raw) const;
 [[nodiscard]] std::uint64_t fingerprint() const noexcept;
 void save(const std::filesystem::path& path) const;
 static InferenceArtifact load(const std::filesystem::path& path);
 static InferenceArtifact read(std::istream& input);
};
}
