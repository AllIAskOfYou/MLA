#pragma once

#include <torch/torch.h>
#include "DenseHead.h"

namespace md {

	class SNetImpl : public torch::nn::Module
	{
	public:
		SNetImpl(
			int64_t s_dim,
			int64_t a_dim,
			std::vector<int64_t> units
		) :
			dense(DenseHead(s_dim + a_dim, s_dim, units))
		{
			//register_module("dense_in", dense_in);
			//register_module("conv", conv);
			register_module("dense", dense);
		}

		at::Tensor forward(at::Tensor s, at::Tensor a) {
			auto x = torch::cat({ s, a }, -1);
			// transpose from ( B, in_dim ) to ( B, out_dim )
			x = dense(x);
			return x;
		}

		DenseHead dense;
	};
	TORCH_MODULE(SNet);

}