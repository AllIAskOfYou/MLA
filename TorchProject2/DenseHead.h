#pragma once 

#include <torch/torch.h>

namespace md {

	class DenseHeadImpl : public torch::nn::Module
	{
	public:
		DenseHeadImpl(
			int64_t in_dim,
			int64_t out_dim,
			std::vector<int64_t> units
		)
		{
			units.push_back(out_dim);
			layers = torch::nn::Sequential(torch::nn::Linear(in_dim, units[0]));
			for (int i = 1; i < units.size(); i++) {
				layers->push_back(torch::nn::ReLU());
				layers->push_back(torch::nn::Linear(units[i-1], units[i]));
			}
			units.pop_back();
			register_module("layers", layers);
		}

		at::Tensor forward(at::Tensor x) {
			x = layers->forward(x);
			return x;
		}

		torch::nn::Sequential layers;
	};
	TORCH_MODULE(DenseHead);

}