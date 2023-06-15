#pragma once

#include <torch/torch.h>
#include "DenseHead.h"
#include "ConvHead.h"
#include "State.h"

namespace md {

	class FNetImpl : public torch::nn::Module
	{
	public:
		FNetImpl(
			int64_t as_n,
			int64_t s_dim,
			//std::vector<int64_t> units_in,
			std::vector<int64_t> units_out
			//std::vector<int64_t> dims,
			//std::vector<bool> pools
		) :
			//dense_in(DenseHead(in_dim, units_in)),
			//conv(ConvHead(units_in[-1], dims, pools)),
			dense_out(DenseHead(2*as_n, s_dim, units_out))
		{
			//register_module("dense_in", dense_in);
			//register_module("conv", conv);
			register_module("dense_out", dense_out);
		}

		at::Tensor forward(State state) {
			at::Tensor as = state.aStates;
			at::Tensor os = state.oStates;
			auto x = torch::cat({as, os}, -1);
			// transpose from ( B, L, in_dim ) to ( B, in_dim )
			x = x.index({ at::indexing::Slice(), -1, at::indexing::Slice() });
			// transpose from ( B, in_dim ) to ( B, out_dim )
			x = dense_out(x);

			return x;
		}

		/*
		at::Tensor forward(at::Tensor x) {
			// transpose from ( B, L, in_dim ) to ( B, L, C )
			x = dense_in(x);
			// transpose from ( B, L, C ) to ( B, C, L )
			x = at::transpose(x, 1, 2);
			// transpose from ( B, C, L ) to ( B, C, L' )
			x = conv(x);
			// transpose from ( B, C, L' ) to ( B, C x L' )
			x = x.flatten(1);
			// transpose from ( B, C x L' ) to ( B, N )
			x = dense_out(x);
			// transpose from ( B, N ) to ( B, out_dim )
			x = lin(x);

			return x;
		}
		*/

		//DenseHead dense_in;
		//ConvHead conv;
		DenseHead dense_out;
	};
	TORCH_MODULE(FNet);

}