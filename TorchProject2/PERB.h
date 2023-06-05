#pragma once

#include "ReplayBuffer.h"
#include "SumTree.h"
#include "MaxTree.h"

class PERB : ReplayBuffer {
public:
	PERB(
		int64_t size,
		int64_t last_n,
		int64_t es_n,
		int64_t as_n,
		float alpha,
		float beta
	);

	void push(
		at::Tensor es, at::Tensor as, at::Tensor os,
		at::Tensor aa, at::Tensor oa, at::Tensor r,
		at::Tensor t
	);

	RBSample sample(int64_t batchSize);


private:
	float alpha, beta;
	MaxTree mTree;
	SumTree sTree;
};