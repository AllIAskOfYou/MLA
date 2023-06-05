#pragma once

#include <torch/torch.h>
#include "DQN.h"
#include "XPA.h"
#include "PESampler.h"

class PDQN : public DQN {
public:
	PDQN(
		ReplayBuffer& rb,
		int64_t batch_size,
		torch::nn::AnyModule qNet,
		torch::nn::AnyModule qNetTarget,
		torch::optim::Optimizer& opt,
		XPA& xpa,
		float gamma,
		float delta,
		int pUpdateWait,
		PESampler& pes
	);

	// push method has to be modifed for the needs of priori. exp. buffer
	void push(
		at::Tensor es,
		at::Tensor as,
		at::Tensor os,
		at::Tensor aa,
		at::Tensor oa,
		at::Tensor r,
		at::Tensor t
	) override;

	// takes one update step on one batch from replay buffer
	void update() override;

private:
	PESampler& pes;
};
