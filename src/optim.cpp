#include "neuralforge/optim.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>
#include <limits>
#include <unordered_set>

namespace neuralforge {

SGD::SGD(float learning_rate) : lr_(learning_rate) {
    if (!(learning_rate > 0.0f) || !std::isfinite(learning_rate)) {
        throw std::invalid_argument("SGD: learning_rate must be finite and > 0");
    }
}

void SGD::step(std::vector<Variable>& parameters) {
    for (Variable& p : parameters) {
        if (p.empty() || !p.requires_grad()) continue;
        if (p.grad().size() == 0) {
            throw std::runtime_error("SGD::step: parameter has empty gradient; call backward() first");
        }
        if (p.grad().shape() != p.data().shape()) {
            throw std::runtime_error("SGD::step: grad shape does not match parameter");
        }
        for (std::size_t i = 0; i < p.data().size(); ++i) {
            const float g = p.grad().at(i);
            if (!std::isfinite(g)) {
                throw std::runtime_error("SGD::step: non-finite gradient");
            }
            const float updated = p.data().at(i) - lr_ * g;
            if (!std::isfinite(updated)) {
                throw std::runtime_error("SGD::step: non-finite parameter update");
            }
            p.data().at(i) = updated;
        }
    }
}

void SGD::zero_grad(std::vector<Variable>& parameters) {
    for (Variable& p : parameters) {
        if (!p.empty()) p.zero_grad();
    }
}

Adam::Adam(float learning_rate, float beta1, float beta2, float eps)
    : lr_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps) {
    if (!(lr_ > 0.0f) || !std::isfinite(lr_)) {
        throw std::invalid_argument("Adam: learning_rate must be finite and > 0");
    }
    if (!(beta1_ >= 0.0f && beta1_ < 1.0f) || !std::isfinite(beta1_)) {
        throw std::invalid_argument("Adam: beta1 must be in [0, 1)");
    }
    if (!(beta2_ >= 0.0f && beta2_ < 1.0f) || !std::isfinite(beta2_)) {
        throw std::invalid_argument("Adam: beta2 must be in [0, 1)");
    }
    if (!(eps_ > 0.0f) || !std::isfinite(eps_)) {
        throw std::invalid_argument("Adam: eps must be finite and > 0");
    }
}

void Adam::step(std::vector<Variable>& parameters) {
 if(t_==std::numeric_limits<std::size_t>::max())throw std::overflow_error("Adam timestep overflow");
 const auto next_t=t_+1;
 const float bc1=1.0f-std::pow(beta1_,static_cast<float>(next_t));
 const float bc2=1.0f-std::pow(beta2_,static_cast<float>(next_t));
 auto next_state=state_;std::vector<std::pair<Variable*,Tensor>> updates;
 std::unordered_set<const void*> seen;
 for(auto& p:parameters) {
  if(p.empty()||!p.requires_grad())continue;
  if(!p.grad().size()||p.grad().shape()!=p.data().shape())throw std::runtime_error("Adam gradient shape mismatch");
  const void* key=p.impl().get();
  if(!seen.insert(key).second)throw std::invalid_argument("Adam duplicate parameter");
  auto& state=next_state[key];
  if(!state.m.size()){state.m=Tensor(p.data().shape(),0.0f);state.v=Tensor(p.data().shape(),0.0f);}
  if(state.m.shape()!=p.data().shape()||state.v.shape()!=p.data().shape())throw std::runtime_error("Adam moment shape mismatch");
  Tensor updated=p.data();
  for(std::size_t i=0;i<updated.size();++i) {
   const float g=p.grad().at(i);
   if(!std::isfinite(g))throw std::runtime_error("Adam non-finite gradient");
   const float m=beta1_*state.m.at(i)+(1-beta1_)*g;
   const float v=beta2_*state.v.at(i)+(1-beta2_)*(g*g);
   const float value=p.data().at(i)-lr_*(m/bc1)/(std::sqrt(v/bc2)+eps_);
   if(!std::isfinite(m)||!std::isfinite(v)||v<0||!std::isfinite(value))
    throw std::runtime_error("Adam non-finite parameter or moment update");
   state.m.at(i)=m;state.v.at(i)=v;updated.at(i)=value;
  }
  updates.emplace_back(&p,std::move(updated));
 }
 // Commit only after every parameter and moment update has been validated.
 for(auto& [parameter,value]:updates)parameter->data()=std::move(value);
 state_.swap(next_state);t_=next_t;
}
AdamSnapshot Adam::snapshot(const std::vector<Variable>& parameters) const {
 AdamSnapshot out{lr_,beta1_,beta2_,eps_,t_,{},{}};
 std::unordered_set<const void*> seen;
 for(const auto& p:parameters) {
  if(p.empty()||!p.requires_grad()||!seen.insert(p.impl().get()).second)
   throw std::invalid_argument("Adam snapshot needs unique trainable parameters");
  const auto it=state_.find(p.impl().get());
  out.first.push_back(it==state_.end()?Tensor(p.data().shape(),0.0f):it->second.m);
  out.second.push_back(it==state_.end()?Tensor(p.data().shape(),0.0f):it->second.v);
 }
 return out;
}
void Adam::restore(const AdamSnapshot& saved,const std::vector<Variable>& parameters) {
 Adam checked(saved.learning_rate,saved.beta1,saved.beta2,saved.epsilon);
 if(saved.first.size()!=parameters.size()||saved.second.size()!=parameters.size())
  throw std::invalid_argument("Adam checkpoint parameter count mismatch");
 for(std::size_t n=0;n<parameters.size();++n) {
  const auto& p=parameters[n];
  if(p.empty()||!p.requires_grad()||saved.first[n].shape()!=p.data().shape()||saved.second[n].shape()!=p.data().shape()||
     checked.state_.count(p.impl().get()))throw std::invalid_argument("Adam checkpoint shape or identity mismatch");
  for(std::size_t i=0;i<p.data().size();++i) {
   const auto m=saved.first[n].at(i),v=saved.second[n].at(i);
   if(!std::isfinite(m)||!std::isfinite(v)||v<0||(saved.timestep==0&&(m!=0||v!=0)))
    throw std::invalid_argument("Adam checkpoint contains invalid moments");
  }
  checked.state_.emplace(p.impl().get(),MomentState{saved.first[n],saved.second[n]});
 }
 checked.t_=saved.timestep;
 *this=std::move(checked);
}

void Adam::zero_grad(std::vector<Variable>& parameters) {
    for (Variable& p : parameters) {
        if (!p.empty()) p.zero_grad();
    }
}

} // namespace neuralforge
