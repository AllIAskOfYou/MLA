#pragma once

#include <torch/torch.h>
#include <iostream>
#include "DTensor.h"
#include "PipeServer.h"

class GameSession {
public:
	GameSession(
		int64_t s_n,
		int64_t a_n,
		int64_t batch_size,
		int64_t last_n,
		torch::nn::AnyModule oaModel,
		std::shared_ptr<torch::optim::Optimizer> oaModelOptimizer
	);

	void start();

private:
	int64_t s_n;
	int64_t a_n;
	int64_t last_n;
	int64_t batch_size;

	torch::nn::AnyModule oaModel;
	
	std::shared_ptr<torch::optim::Optimizer> oaModelOptimizer;

	torch::nn::CrossEntropyLoss ceLoss;

	DTensor states;
	DTensor aActions;
	DTensor oActions;
	DTensor rewards;

	torch::Tensor s;
	torch::Tensor oa;
	torch::Tensor r;
	torch::Tensor next_aa;
	torch::Tensor next_oa;

	int readS, readA, readR;

	torch::TensorOptions floatOptions = torch::TensorOptions().dtype(torch::kFloat32);
	torch::TensorOptions intOptions = torch::TensorOptions().dtype(torch::kInt64);

	PipeServer ps;
};