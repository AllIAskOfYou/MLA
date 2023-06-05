#include "PERB.h"

PERB::PERB(
	int64_t size,
	int64_t last_n,
	int64_t es_n,
	int64_t as_n,
	float alpha,
	float beta
) :
	ReplayBuffer(size, last_n, es_n, as_n),
	alpha(alpha),
	beta(beta),
	mTree(size),
	sTree(size)
{}

void PERB::push(
	at::Tensor es, at::Tensor as, at::Tensor os,
	at::Tensor aa, at::Tensor oa, at::Tensor r,
	at::Tensor t
)
{
	ReplayBuffer::push(es, as, os, aa, oa, r, t);
	float maxPe = mTree.get_value();
	maxPe = maxPe == 0 ? 1 : maxPe;
	mTree.push(maxPe);
	sTree.push(maxPe);
}

RBSample PERB::sample(int64_t batchSize) {
	auto indexes = sTree.sample_batch(batchSize);
	at::Tensor idx = at::from_blob(indexes.data(), { batchSize });
	return get_sample(idx);
}