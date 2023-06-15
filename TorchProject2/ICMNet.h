#pragma once

#include <torch/torch.h>
#include <chrono>
#include "DenseHead.h"
#include "ConvHead.h"
#include "FNet.h"
#include "ANet.h"
#include "SNet.h"
#include "RBSample.h"

struct ICMReturn {
	at::Tensor aPred;
	at::Tensor sNext;
	at::Tensor sNextPred;
};

namespace md {

	class ICMNetImpl : public torch::nn::Module
	{
	public:
		ICMNetImpl(
			int64_t es_n,
			int64_t as_n,
			int64_t a_n,
			int64_t last_n,
			int64_t a_emb,
			int64_t s_emb,
			std::vector<int64_t> units_f,
			std::vector<int64_t> units_a,
			std::vector<int64_t> units_s
		) :
			emb_a(torch::nn::Embedding(a_n, a_emb)),
			fnet(FNet(as_n, s_emb, units_f)),
			anet(ANet(s_emb, a_emb, units_a)),
			snet(SNet(s_emb, a_emb, units_s))
		{
			register_module("emb_a", emb_a);
			register_module("fnet", fnet);
			register_module("anet", anet);
			register_module("snet", snet);
		}

		ICMReturn forward(
			RBSample smpl
		)
		{

			// extract features from states
			auto s = fnet(smpl.states);
			auto s_next = fnet(smpl.nStates);

			// inverse module
			auto a_pred = anet(s, s_next);

			// embed actions
			auto a_emb = emb_a(smpl.aActions);
			auto s_next_pred = snet(s, a_emb);

			ICMReturn out;
			out.aPred = a_pred;
			out.sNext = s_next;
			out.sNextPred = s_next_pred;

			return out;
		}

		torch::nn::Embedding emb_a;
		FNet fnet;
		ANet anet;
		SNet snet;
	};
	TORCH_MODULE(ICMNet);
	
}
