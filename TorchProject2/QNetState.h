#pragma once

#include <torch/torch.h>

namespace md {

	class QNetState : public torch::nn::Module {
	public:
		QNetState(int64_t s_n, int64_t a_n, int64_t last_n) :
			lin(torch::nn::Linear(s_n* last_n, 8)),
			lin2(torch::nn::Linear(8, a_n))
		{
			register_module("lin", lin);
			register_module("lin2", lin2);
		}

		at::Tensor forward(at::Tensor s, at::Tensor aa, at::Tensor oa) {
			auto x = lin(s.flatten(1));
			x = torch::relu(x);
			x = lin2(x);
			return x;
		}

		torch::nn::Linear lin;
		torch::nn::Linear lin2;
	};

}