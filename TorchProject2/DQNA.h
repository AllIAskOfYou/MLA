#pragma once

#include <torch/torch.h>
#include "DQN.h"
#include "XPA.h"
#include "ICMReturn.h"

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
		torch::nn::AnyModule iCMNet,
		float lambda,
		float beta,
		float ro
	);

	// takes one update step on one batch from replay buffer
	void update() override;

private:
	torch::nn::AnyModule iCMNet;
	// extrinsic update factor
	float lambda;
	// inverse vs forward module update ratio
	float beta;
	// intrinsic reward factor
	float ro;
};
