#pragma once

#include <torch/torch.h>
#include "DQN.h"
#include "XPA.h"

class DQNA : public DQN {
public:
	DQNA(
		ReplayBuffer& rb,
		int64_t batch_size,
		torch::nn::AnyModule qNet,
		torch::nn::AnyModule qNetTarget,
		torch::optim::Optimizer& opt,
		torch::optim::LRScheduler& lrs,
		XPA& xpa,
		float gamma,
		float delta,
		int pUpdateWait,
		torch::nn::AnyModule oaNet,
		//torch::optim::Optimizer& oaNetOpt,
		//torch::optim::LRScheduler& oaNetLrs,
		float lambda,
		float ro
	);

	// takes one update step on one batch from replay buffer
	void update() override;

private:
	torch::nn::AnyModule oaNet;
	//torch::optim::Optimizer& oaNetOpt;
	//torch::optim::LRScheduler& oaNetLrs;
	// extrinsic update factor
	float lambda;
	// intrinsic reward factor
	float ro;
};
