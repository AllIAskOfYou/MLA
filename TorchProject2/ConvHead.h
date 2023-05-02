#pragma once 

#include <torch/torch.h>
#include "ConvBlock.h"

class ConvHeadImpl : public torch::nn::Module
{
public:
	ConvHeadImpl(int64_t in_dim, std::vector<int64_t> dims, std::vector<bool> pools)
	{
		layers = torch::nn::Sequential(ConvBlock(in_dim, dims[0], pools[0]));
		for (int i = 1; i < dims.size(); i++) {
			layers->push_back(ConvBlock(dims[i-1], dims[i], pools[i]));
		}
		register_module("layers", layers);
	}

	at::Tensor forward(at::Tensor x) {
		x = layers->forward(x);
		return x;
	}
	
	torch::nn::Sequential layers;
};
TORCH_MODULE(ConvHead);