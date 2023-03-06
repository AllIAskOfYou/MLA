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
		torch::nn::AnyModule envModel,
		std::shared_ptr<torch::optim::Optimizer> oaModelOptimizer,
		std::shared_ptr<torch::optim::Optimizer> envModelOptimizer
	);

	void start();

private:
	void updateOaModel();
	void updateEnvModel();
	torch::Tensor getNextOa();
	torch::Tensor decideNextAction();
	//torch::Tensor getNextS(torch::Tensor s, torch::Tensor aa, torch::Tensor oa);

private:
	int64_t s_n;
	int64_t a_n;
	int64_t last_n;
	int64_t batch_size;

	torch::nn::AnyModule oaModel;
	torch::nn::AnyModule envModel;
	
	std::shared_ptr<torch::optim::Optimizer> oaModelOptimizer;
	std::shared_ptr<torch::optim::Optimizer> envModelOptimizer;

	torch::nn::CrossEntropyLoss ceLoss;
	torch::nn::L1Loss l1Loss;

	DTensor states;
	DTensor aActions;
	DTensor oActions;

	torch::Tensor s;
	torch::Tensor oa;
	torch::Tensor next_aa;
	torch::Tensor next_oa;

	int readS, readA;

	torch::TensorOptions floatOptions = torch::TensorOptions().dtype(torch::kFloat32);
	torch::TensorOptions intOptions = torch::TensorOptions().dtype(torch::kInt64);

	PipeServer ps;
};