#pragma once

#include <torch/torch.h>

namespace md {

	class QNet : public torch::nn::Module {
	public:
		QNet(int64_t s_n, int64_t a_n, int64_t last_n) :
			lin(torch::nn::Linear(s_n* last_n, 8)),
			emb(torch::nn::Embedding(a_n, 8)),
			lin_emb(torch::nn::Linear(last_n * 8, 8)),
			out(torch::nn::Linear(16, a_n))
		{
			register_module("lin", lin);
			register_module("emb", emb);
			register_module("lin_emb", lin_emb);
		}

		at::Tensor forward(at::Tensor s, at::Tensor oa) {
			//s = s.transpose(1, 2);
			/*
			std::cout << s.sizes() << std::endl;
			std::cout << oa.sizes() << std::endl;
			std::cout << s.flatten(1).sizes() << std::endl;
			std::cout << oa.flatten(1).sizes() << std::endl;
			std::cout << lin(s.flatten(1)) << std::endl;
			std::cout << emb(oa.flatten(1)) << std::endl;
			*/
			return out(flatten(torch::cat({ lin(s.flatten(1)), lin_emb(emb(oa.flatten(1)).flatten(1)) }, -1)));
		}

		torch::nn::Linear lin;
		torch::nn::Embedding emb;
		torch::nn::Linear lin_emb;
		torch::nn::Linear out;
		torch::nn::Flatten flatten;
	};

}