#include "neuralforge/model.hpp"
#include "neuralforge/nn_ops.hpp"

#include <cmath>
#include <algorithm>
#include <bit>
#include <cstdint>
#include "neuralforge/kernels.hpp"
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace neuralforge {
namespace {

// Guards against hostile or corrupted model headers that would attempt to
// allocate multi-gigabyte weight tensors before any parameter is read.
constexpr std::size_t kMaxFeatureDim = 65'536;           // per axis
constexpr std::size_t kMaxParameterElements = 16'777'216; // ~64 MiB of float

[[nodiscard]] bool checked_mul_size(std::size_t a, std::size_t b, std::size_t& out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

std::size_t read_dimension(std::istream& in, const char* label) {
    long long value{};
    if (!(in >> value) || value <= 0 ||
        static_cast<unsigned long long>(value) > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string("Invalid model dimension: ") + label);
    }
    const auto dim = static_cast<std::size_t>(value);
    if (dim > kMaxFeatureDim) {
        throw std::runtime_error(std::string("Model dimension exceeds safety limit: ") + label);
    }
    return dim;
}

void enforce_parameter_budget(std::size_t input, std::size_t hidden, std::size_t output) {
    std::size_t w1 = 0, w2 = 0, total = 0;
    if (!checked_mul_size(input, hidden, w1) || !checked_mul_size(hidden, output, w2)) {
        throw std::runtime_error("Model parameter count overflows size_t");
    }
    // total = w1 + hidden + w2 + output
    if (w1 > kMaxParameterElements || w2 > kMaxParameterElements) {
        throw std::runtime_error("Model weight matrix exceeds safety limit");
    }
    total = w1;
    if (total > kMaxParameterElements - hidden) {
        throw std::runtime_error("Model parameter count exceeds safety limit");
    }
    total += hidden;
    if (total > kMaxParameterElements - w2) {
        throw std::runtime_error("Model parameter count exceeds safety limit");
    }
    total += w2;
    if (total > kMaxParameterElements - output) {
        throw std::runtime_error("Model parameter count exceeds safety limit");
    }
}

void read_values(std::istream& in, Tensor& tensor, const char* label) {
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        float value{};
        if (!(in >> value) || !std::isfinite(value)) {
            throw std::runtime_error(std::string("Missing or non-finite model parameter: ") + label);
        }
        tensor.at(i) = value;
    }
}

void require_finite_inputs(const Tensor& inputs) {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (!std::isfinite(inputs.at(i))) {
            throw std::invalid_argument("Model input contains non-finite values");
        }
    }
}

} // namespace

MLPModel MLPModel::load(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open model file: " + path.string());
    return read(in);
}

MLPModel MLPModel::read(std::istream& in) {
    std::string magic;
    int version{};
    if (!(in >> magic >> version) || magic != "NEURALFORGE_MLP" || version != 1) {
        throw std::runtime_error("Unsupported or malformed NeuralForge model header");
    }

    MLPModel model;
    model.input_features_ = read_dimension(in, "input");
    model.hidden_features_ = read_dimension(in, "hidden");
    model.output_features_ = read_dimension(in, "output");
    enforce_parameter_budget(model.input_features_, model.hidden_features_, model.output_features_);

    model.w1_ = Tensor({model.input_features_, model.hidden_features_});
    model.b1_ = Tensor({model.hidden_features_});
    model.w2_ = Tensor({model.hidden_features_, model.output_features_});
    model.b2_ = Tensor({model.output_features_});
    read_values(in, model.w1_, "w1");
    read_values(in, model.b1_, "b1");
    read_values(in, model.w2_, "w2");
    read_values(in, model.b2_, "b2");
    std::string trailing;
    if (in >> trailing) throw std::runtime_error("Unexpected trailing data in model file");
    return model;
}

Tensor MLPModel::predict_proba(const Tensor& inputs) const {
    if (inputs.ndim() != 2 || inputs.shape()[1] != input_features_) {
        throw std::invalid_argument("Model input must have shape (batch, " +
                                    std::to_string(input_features_) + ")");
    }
    // batch == 0 is valid: returns an empty (0, C) probability tensor.
    require_finite_inputs(inputs);
    return softmax_rows(dense(relu(dense(inputs, w1_, b1_)), w2_, b2_));
}

std::vector<std::size_t> MLPModel::predict(const Tensor& inputs) const {
    const Tensor probabilities = predict_proba(inputs);
    std::vector<std::size_t> result(probabilities.shape()[0]);
    for (std::size_t row = 0; row < probabilities.shape()[0]; ++row) {
        std::size_t best = 0;
        for (std::size_t col = 1; col < output_features_; ++col) {
            if (probabilities.at(row, col) > probabilities.at(row, best)) best = col;
        }
        result[row] = best;
    }
    return result;
}

MLPModel MLPModel::from_parameters(const std::array<Tensor,4>& p) {
 if(p[0].ndim()!=2||p[1].ndim()!=1||p[2].ndim()!=2||p[3].ndim()!=1)
  throw std::invalid_argument("MLP parameter ranks are invalid");
 const auto in=p[0].shape()[0],h=p[0].shape()[1],out=p[2].shape()[1];
 if(!in||!h||!out||in>kMaxFeatureDim||h>kMaxFeatureDim||out>kMaxFeatureDim||
    p[1].size()!=h||p[2].shape()[0]!=h||p[3].size()!=out)
  throw std::invalid_argument("MLP parameter dimensions are invalid");
 enforce_parameter_budget(in,h,out);
 for(const auto& t:p)for(std::size_t i=0;i<t.size();++i)
  if(!std::isfinite(t.data()[i]))throw std::invalid_argument("MLP contains non-finite parameters");
 MLPModel m;m.input_features_=in;m.hidden_features_=h;m.output_features_=out;
 m.w1_=p[0];m.b1_=p[1];m.w2_=p[2];m.b2_=p[3];return m;
}
InferenceWorkspace MLPModel::prepare() const {
 if(!input_features_||!hidden_features_||!output_features_)throw std::invalid_argument("Uninitialized model");
 InferenceWorkspace w;w.input_=input_features_;w.hidden_=hidden_features_;w.output_=output_features_;
 w.scratch_.resize(hidden_features_);return w;
}
void MLPModel::predict_proba_into(std::span<const float> x,std::size_t batch,
                                  std::span<float> y,InferenceWorkspace& w) const {
 std::size_t nx=0,ny=0;
 if(!input_features_||!hidden_features_||!output_features_||
    !checked_mul_size(batch,input_features_,nx)||!checked_mul_size(batch,output_features_,ny)||
    x.size()!=nx||y.size()!=ny)throw std::invalid_argument("Inference buffer sizes do not match model and batch");
 if(w.input_!=input_features_||w.hidden_!=hidden_features_||w.output_!=output_features_||w.scratch_.size()!=hidden_features_)
  throw std::invalid_argument("Workspace does not match model");
 if(nx && ny) {
  const auto a=reinterpret_cast<std::uintptr_t>(x.data()),b=reinterpret_cast<std::uintptr_t>(y.data());
  if(nx>std::numeric_limits<std::uintptr_t>::max()/sizeof(float)||ny>std::numeric_limits<std::uintptr_t>::max()/sizeof(float))
   throw std::invalid_argument("Inference byte count overflow");
  const auto ax=nx*sizeof(float),by=ny*sizeof(float);
  if(a>std::numeric_limits<std::uintptr_t>::max()-ax||b>std::numeric_limits<std::uintptr_t>::max()-by||
     (a<b+by&&b<a+ax))throw std::invalid_argument("Inference input and output buffers overlap");
 }
 for(float value:x)if(!std::isfinite(value))throw std::invalid_argument("Non-finite inference input");
 const auto axpy=selected_axpy();
 auto* hidden=w.scratch_.data();
 for(std::size_t row=0;row<batch;++row) {
  std::copy_n(b1_.data(),hidden_features_,hidden);
  for(std::size_t k=0;k<input_features_;++k)axpy(hidden,w1_.data()+k*hidden_features_,x[row*input_features_+k],hidden_features_);
  for(std::size_t j=0;j<hidden_features_;++j) {
   if(!std::isfinite(hidden[j]))throw std::runtime_error("Non-finite hidden activation");
   hidden[j]=std::max(0.0f,hidden[j]);
  }
  auto* out=y.data()+row*output_features_;std::copy_n(b2_.data(),output_features_,out);
  for(std::size_t k=0;k<hidden_features_;++k)axpy(out,w2_.data()+k*output_features_,hidden[k],output_features_);
  float maximum=-std::numeric_limits<float>::infinity();
  for(std::size_t j=0;j<output_features_;++j) {
   if(!std::isfinite(out[j]))throw std::runtime_error("Non-finite inference logit");
   maximum=std::max(maximum,out[j]);
  }
  double sum=0;
  for(std::size_t j=0;j<output_features_;++j) {out[j]=static_cast<float>(std::exp(static_cast<double>(out[j])-maximum));sum+=out[j];}
  for(std::size_t j=0;j<output_features_;++j)out[j]=static_cast<float>(out[j]/sum);
 }
}
std::uint64_t MLPModel::fingerprint() const noexcept {
 std::uint64_t hash=14695981039346656037ULL;
 const auto feed=[&](std::uint64_t value,int bytes){for(int i=0;i<bytes;++i){hash^=(value>>(8*i))&255U;hash*=1099511628211ULL;}};
 feed(input_features_,8);feed(hidden_features_,8);feed(output_features_,8);
 for(const auto* t:{&w1_,&b1_,&w2_,&b2_})for(std::size_t i=0;i<t->size();++i)feed(std::bit_cast<std::uint32_t>(t->data()[i]),4);
 return hash;
}

} // namespace neuralforge
