#pragma once

#include <torch/torch.h>

namespace md {

	class QNetBasic : public torch::nn::Module {
	public:
		QNetBasic(int64_t s_n, int64_t a_n, int64_t last_n, int64_t a_emb) :
			lin(torch::nn::Linear(s_n* last_n, 8)),
			emb(a_n, a_emb),
			lin_aa(torch::nn::Linear(last_n * a_emb, 8)),
			lin_oa(torch::nn::Linear(last_n* a_emb, 8)),
			lin_out(torch::nn::Linear(3 * 8, a_n))
		{
			register_module("lin", lin);
			register_module("emb", emb);
			register_module("lin_aa", lin_aa);
			register_module("lin_oa", lin_oa);
			register_module("lin_out", lin_out);
		}

		at::Tensor forward(at::Tensor s, at::Tensor aa, at::Tensor oa) {
			auto s_emb = lin(s.flatten(1));
			auto aa_emb = emb(aa);
			aa_emb = lin_aa(aa_emb.flatten(1));
			auto oa_emb = emb(oa);
			oa_emb = lin_oa(oa_emb.flatten(1));
			auto x = torch::cat({ s_emb, aa_emb, oa_emb }, -1);
			x = lin_out(x);
			return x;
		}

		torch::nn::Linear lin;
		torch::nn::Embedding emb;
		torch::nn::Linear lin_aa;
		torch::nn::Linear lin_oa;
		torch::nn::Linear lin_out;
	};

}