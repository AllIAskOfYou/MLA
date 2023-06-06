#pragma once

#include <torch/torch.h>
#include "RLA.h"
#include "XPA.h"

class DQN : public RLA {
public:
	DQN(
		ReplayBuffer& rb,
		int64_t batch_size,
		torch::nn::AnyModule qNet,
		torch::nn::AnyModule qNetTarget,
		torch::optim::Optimizer& opt,
		torch::optim::LRScheduler& lrs,
		XPA& xpa,
		float gamma,
		float delta,
		int pUpdateWait
	);

	// takes one update step on one batch from replay buffer
	void update();

	// returns the next action to be preformed by the agent
	at::Tensor nextAction();

	// returns the next action to be preformed by the self oponent
	at::Tensor selfPlay();

	ReplayBuffer& get_rb() {
		return rb;
	}

protected:
	at::Tensor calculate_err(RBSample rs);
	void update_params(float delta);

protected:
	torch::nn::AnyModule qNet, qNetTarget;
	torch::optim::Optimizer& opt;
	torch::optim::LRScheduler& lrs;
	XPA& xpa;
	float gamma, delta;
	int pUpdateWait;
	int pUpdateTimes = 0;
};

