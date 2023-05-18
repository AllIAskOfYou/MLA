#pragma once

#include <torch/torch.h>
#include <chrono>
#include "ConvHead.h"
#include "State.h"

namespace md {

	class QNetConvImpl : public torch::nn::Module
	{
	public:
		QNetConvImpl(
			int64_t es_n,
			int64_t as_n,
			int64_t a_n,
			int64_t last_n,
			int64_t es_emb,
			int64_t as_emb,
			int64_t a_emb,
			std::vector<int64_t> dims,
			std::vector<bool> pools,
			std::vector<int64_t> units
		) :
			bn_es(torch::nn::BatchNorm1d(es_n)),
			bn_as(torch::nn::BatchNorm1d(as_n)),
			lin_es(torch::nn::Linear(es_n, es_emb)),
			lin_as(torch::nn::Linear(as_n, as_emb)),
			emb(torch::nn::Embedding(a_n, a_emb)),
			conv_es(ConvHead(es_emb, dims, pools)),
			conv_as(ConvHead(as_n, dims, pools)),
			conv_a(ConvHead(a_emb, dims, pools)),
			out(torch::nn::Sequential())
		{
			int64_t l(1);
			for (int i = 0; i < pools.size(); i++) {
				if (pools[i]) {
					l *= 2;
				}
			}

			out->push_back(torch::nn::Linear(1 * dims[dims.size() - 1] * (last_n / l), units[0]));
			for (int i = 1; i < units.size(); i++) {
				out->push_back(torch::nn::ReLU());
				out->push_back(torch::nn::Linear(units[i - 1], units[i]));
			}

			register_module("bn_es", bn_es);
			register_module("bn_as", bn_as);
			register_module("lin_es", lin_es);
			register_module("lin_as", lin_as);
			register_module("emb", emb);
			register_module("conv_es", conv_es);
			register_module("conv_as", conv_as);
			register_module("conv_a", conv_a);
			register_module("out", out);
		}

		at::Tensor forward(
			State state
		) 
		{
			//at::Tensor es = state.eStates;
			at::Tensor as = state.aStates;
			//at::Tensor os = state.oStates;
			//at::Tensor aa = state.aActions;
			//at::Tensor oa = state.oActions;

			//auto t0 = std::chrono::high_resolution_clock::now();

			// normalize states representation per feature

			//es = at::transpose(es, 1, 2);
			//as = at::transpose(as, 1, 2);
			//os = at::transpose(os, 1, 2);
			//es = bn_es(es);
			//as = bn_as(as);
			//os = bn_as(os);
			//es = at::transpose(es, 1, 2);
			//as = at::transpose(as, 1, 2);
			//os = at::transpose(os, 1, 2);

			// linearly transform ( B, L, s_n) -> ( B, L, s_emb)
			//es = lin_es(es);
			//as = lin_as(as);
			//os = lin_as(os);

			// linearly transform ( B, L ) -> ( B, L, C ) 
			//aa = emb(aa);
			//oa = emb(oa);

			// transpose from ( B, L, C ) to ( B, C, L )
			//es = at::transpose(es, 1, 2);
			as = at::transpose(as, 1, 2);
			//os = at::transpose(os, 1, 2);
			//aa = at::transpose(aa, 1, 2);
			//oa = at::transpose(oa, 1, 2);

			//auto t1 = std::chrono::high_resolution_clock::now();
			// conv
			//es = conv_es(es);
			as = conv_as(as);
			//os = conv_as(os);
			//aa = conv_a(aa);
			//oa = conv_a(oa);
			//auto t2 = std::chrono::high_resolution_clock::now();

			//es = es.flatten(1);
			as = as.flatten(1);
			//os = os.flatten(1);
			//aa = aa.flatten(1);
			//oa = oa.flatten(1);

			at::Tensor x = torch::cat({ as }, -1);

			//std::cout << "X: \n" << x << std::endl;
			x = out->forward(x);
			//auto t3 = std::chrono::high_resolution_clock::now();

			//auto d0 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
			//auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t0);
			//std::cout << "full time: \n" << d1.count() << "ms\nconv: \n" << d0.count() << std::endl;
			return x;
		}

		torch::nn::BatchNorm1d bn_es;
		torch::nn::BatchNorm1d bn_as;
		torch::nn::Linear lin_es;
		torch::nn::Linear lin_as;
		torch::nn::Embedding emb;
		ConvHead conv_es;
		ConvHead conv_as;
		ConvHead conv_a;
		torch::nn::Sequential out;
	};
	TORCH_MODULE(QNetConv);

}