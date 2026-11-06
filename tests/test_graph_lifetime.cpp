#include <gtest/gtest.h>
#include "neuralforge/autograd.hpp"
#include "neuralforge/loss.hpp"
using namespace neuralforge;
TEST(GraphLifetime, OperationsReleaseOutputsAfterBackward) {
 Variable x(Tensor{{1,2}},true),w(Tensor{{1,0},{0,1}},true),bias(Tensor({2},0),true);
 std::weak_ptr<Variable::Impl> hidden,logits,loss_ref;
 for(int i=0;i<100;++i) {
  {auto h=relu(add(matmul(x,w),bias));hidden=h.impl();
   auto y=add(h,bias);logits=y.impl();auto loss=cross_entropy_loss(y,{1});loss_ref=loss.impl();loss.backward();}
  EXPECT_TRUE(hidden.expired());EXPECT_TRUE(logits.expired());EXPECT_TRUE(loss_ref.expired());
  x.zero_grad();w.zero_grad();bias.zero_grad();
 }
}
TEST(GraphLifetime, MseAndSumDoNotRetainOwnOutputs) {
 Variable x(Tensor{{1,2}},true),target(Tensor{{0,0}},false);
 std::weak_ptr<Variable::Impl> mse,total;
 {auto value=mse_loss(x,target);mse=value.impl();value.backward();}
 {auto value=sum(x);total=value.impl();value.backward();}
 EXPECT_TRUE(mse.expired());EXPECT_TRUE(total.expired());
}
