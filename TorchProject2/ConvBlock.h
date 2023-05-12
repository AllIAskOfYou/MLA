#pragma once

#include <torch/torch.h>

namespace md {

class ConvBlockImpl: public torch::nn::Module {
public:
	ConvBlockImpl(int64_t in_dim, int64_t out_dim, int64_t do_max_pool) :
		conv(torch::nn::Conv1d(
			torch::nn::Conv1dOptions(in_dim, out_dim, 3).padding(1)
		)),
		relu(torch::nn::ReLU()),
		batch_norm(torch::nn::BatchNorm1d(out_dim)),
		max_pool(torch::nn::MaxPool1d(2)),
		do_max_pool(do_max_pool)
	{
		register_module("conv", conv);
		register_module("relu", relu);
		register_module("batch_norm", batch_norm);
		register_module("max_pool", max_pool);
	}

	at::Tensor forward(at::Tensor x) {
		x = conv(x);
		x = relu(x);
		x = batch_norm(x);
		if (do_max_pool == true) {
			x = max_pool(x);
		}
		return x;
	}

	torch::nn::Conv1d conv;
	torch::nn::ReLU relu;
	torch::nn::BatchNorm1d batch_norm;
	torch::nn::MaxPool1d max_pool;

	bool do_max_pool = false;
};
TORCH_MODULE(ConvBlock);

}