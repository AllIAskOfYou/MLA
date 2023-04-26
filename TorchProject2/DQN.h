#pragma once

#include "RLA.h"
#include <torch/torch.h>

class DQN : public RLA {
public:
	DQN(
		ReplayBuffer& rb,
		int64_t batch_size,
		torch::nn::AnyModule qNet,
		torch::nn::AnyModule qNetTarget,
		torch::optim::Optimizer& opt,
		float gamma,
		float delta
	);

	// takes one update step on one batch from replay buffer
	void update();

	// returns the next action to be preformed by the agent
	at::Tensor nextAction();

	ReplayBuffer& get_rb() {
		return rb;
	}

private:
	torch::nn::AnyModule qNet, qNetTarget;
	torch::optim::Optimizer& opt;
	float gamma, delta;
};

