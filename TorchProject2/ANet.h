#pragma once

#include <torch/torch.h>
#include "DenseHead.h"

namespace md {

	class ANetImpl : public torch::nn::Module
	{
	public:
		ANetImpl(
			int64_t s_dim,
			int64_t a_dim,
			std::vector<int64_t> units
		) :
			dense(DenseHead(2 * s_dim, a_dim, units))
		{
			//register_module("dense_in", dense_in);
			//register_module("conv", conv);
			register_module("dense", dense);
		}

		at::Tensor forward(at::Tensor s, at::Tensor s_next) {
			auto x = torch::cat({ s, s_next }, -1);
			// transpose from ( B, in_dim ) to ( B, out_dim )
			x = dense(x);
			return x;
		}

		DenseHead dense;
	};
	TORCH_MODULE(ANet);

}