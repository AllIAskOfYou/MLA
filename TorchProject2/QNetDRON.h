#pragma once

#include <torch/torch.h>
#include <chrono>
#include "ConvHead.h"
#include "State.h"

namespace md {

	class QNetDRONImpl : public torch::nn::Module
	{
	public:
		QNetDRONImpl(
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
			emb(torch::nn::Embedding(a_n, a_emb)),
			conv_as(ConvHead(as_n, dims, pools)),
			conv_a(ConvHead(a_emb, dims, pools)),
			out(torch::nn::Sequential()),
			weights(torch::nn::Sequential())
		{
			int64_t l(1);
			for (int i = 0; i < pools.size(); i++) {
				if (pools[i]) {
					l *= 2;
				}
			}

			out->push_back(torch::nn::Linear(2 * dims[dims.size() - 1] * (last_n / l), units[0]));
			for (int i = 1; i < units.size(); i++) {
				out->push_back(torch::nn::ReLU());
				out->push_back(torch::nn::Linear(units[i - 1], units[i]));
			}

			weights->push_back(torch::nn::Linear(2 * dims[dims.size() - 1] * (last_n / l), 4));
			weights->push_back(torch::nn::ReLU());


			register_module("emb", emb);
			register_module("conv_as", conv_as);
			register_module("conv_a", conv_a);
			register_module("out", out);
			register_module("weights", weights);
		}

		at::Tensor forward(
			State state
		)
		{
			//at::Tensor es = state.eStates;
			at::Tensor as = state.aStates;
			at::Tensor os = state.oStates;
			at::Tensor aa = state.aActions;
			at::Tensor oa = state.oActions;

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
			aa = emb(aa);
			oa = emb(oa);

			// transpose from ( B, L, C ) to ( B, C, L )
			//es = at::transpose(es, 1, 2);
			as = at::transpose(as, 1, 2);
			os = at::transpose(os, 1, 2);
			aa = at::transpose(aa, 1, 2);
			oa = at::transpose(oa, 1, 2);

			//auto t1 = std::chrono::high_resolution_clock::now();
			// conv
			//es = conv_es(es);
			as = conv_as(as);
			os = conv_as(os);
			aa = conv_a(aa);
			oa = conv_a(oa);
			//auto t2 = std::chrono::high_resolution_clock::now();

			//es = es.flatten(1);
			as = as.flatten(1);
			os = os.flatten(1);
			aa = aa.flatten(1);
			oa = oa.flatten(1);

			at::Tensor x = torch::cat({ as, aa }, -1);
			//at::Tensor x = torch::cat({ as, oa }, -1);

			//std::cout << "X: \n" << x << std::endl;
			x = out->forward(x);
			//std::cout << "1: " << x.sizes() << "\n" << x << std::endl;
			x = x.view({x.size(0), 4, -1});
			//std::cout << "2: " << x.sizes() << "\n" << x << std::endl;

			at::Tensor w = torch::cat({ os, oa }, -1);
			w = weights->forward(w);
			//std::cout << "3: " << w.sizes() << "\n" << x << std::endl;
			w = w.softmax(-1);
			x = w.matmul(x);
			x = x.view({ x.size(0), -1 });
			//std::cout << "4: " << x.sizes() << "\n" << x << std::endl;

			//auto t3 = std::chrono::high_resolution_clock::now();

			//auto d0 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
			//auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t0);
			//std::cout << "full time: \n" << d1.count() << "ms\nconv: \n" << d0.count() << std::endl;
			return x;
		}

		torch::nn::Embedding emb;
		ConvHead conv_as;
		ConvHead conv_a;
		torch::nn::Sequential out;
		torch::nn::Sequential weights;
	};
	TORCH_MODULE(QNetDRON);

}