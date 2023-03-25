#pragma once

#include "RLA.h"
#include <torch/torch.h>

class DQN : public RLA {
public:
	DQN(
		ReplayBuffer& rb,
		int64_t batch_size,
		torch::optim::Optimizer& opt,
		float gamma,
		float delta
	);

	void update();
	at::Tensor nextAction();

private:
	torch::nn::AnyModule qNet;
	torch::nn::AnyModule qNetTarget;
	torch::optim::Optimizer& opt;
	float gamma;
	float delta;
};

