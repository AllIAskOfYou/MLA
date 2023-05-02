#pragma once

#include <torch/torch.h>
#include "ConvHead.h"
#include <chrono>

namespace md {

	class QNetConvImpl : public torch::nn::Module {
	public:
		QNetConvImpl(
			int64_t s_n,
			int64_t a_n,
			int64_t last_n,
			int64_t s_emb,
			int64_t a_emb,
			std::vector<int64_t> dims,
			std::vector<bool> pools,
			std::vector<int64_t> units
		) :
			lin(torch::nn::Linear(s_n, s_emb)),
			emb(torch::nn::Embedding(a_n, a_emb)),
			bn(torch::nn::BatchNorm1d(s_emb)),
			conv_s(ConvHead(s_emb, dims, pools)),
			conv_a(ConvHead(a_emb, dims, pools)),
			out(torch::nn::Sequential())
		{
			int64_t l(1);
			for (int i = 0; i < pools.size(); i++) {
				if (pools[i]) {
					l *= 2;
				}
			}
			
			out->push_back(torch::nn::Linear(3 * dims[dims.size() - 1] * (last_n / l), units[0]));
			for (int i = 1; i < units.size(); i++) {
				out->push_back(torch::nn::ReLU());
				out->push_back(torch::nn::Linear(units[i-1], units[i]));
			}

			register_module("lin", lin);
			register_module("emb", emb);
			register_module("conv_s", conv_s);
			register_module("conv_a", conv_a);
			register_module("out", out);
		}

		at::Tensor forward(at::Tensor s, at::Tensor aa, at::Tensor oa) {
			auto t0 = std::chrono::high_resolution_clock::now();
			// linearly transform ( B, L, s_n) -> ( B, L, s_emb)
			s = lin(s);

			// linearly transform ( B, L ) -> ( B, L, C ) 
			aa = emb(aa);
			oa = emb(oa);

			// transpose from ( B, L, C ) to ( B, C, L )
			s = at::transpose(s, 1, 2);
			aa = at::transpose(aa, 1, 2);
			oa = at::transpose(oa, 1, 2);

			// normalize states representation per feature
			s = bn(s);

			auto t1 = std::chrono::high_resolution_clock::now();
			// conv
			s = conv_s(s);
			aa = conv_a(aa);
			oa = conv_a(oa);
			auto t2 = std::chrono::high_resolution_clock::now();


			s = s.flatten(1);
			aa = aa.flatten(1);
			oa = oa.flatten(1);

			at::Tensor x = torch::cat({ s, aa, oa }, -1);

			x = out->forward(x);
			auto t3 = std::chrono::high_resolution_clock::now();

			auto d0 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
			auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t0);
			std::cout << "full time: \n" << d1.count() << "ms\nconv: \n" << d0.count() << std::endl;
			return x;
		}
		torch::nn::Linear lin;
		torch::nn::Embedding emb;
		torch::nn::BatchNorm1d bn;
		ConvHead conv_s;
		ConvHead conv_a;
		torch::nn::Sequential out;
	};
	TORCH_MODULE(QNetConv);

}