#include "PESampler.h"
#include <iostream>

PESampler::PESampler(
	int64_t buffer_size,
	float alpha,
	float beta
) :
	buffer_size(buffer_size),
	alpha(alpha),
	beta(beta),
	sTree(buffer_size),
	mTree(buffer_size)
{}

void PESampler::push() {
	float maxPe = mTree.get_value();
	maxPe = (maxPe == 0) ? 1 : maxPe;
	std::cout << "maxPe: " << maxPe << std::endl;
	std::cout << "TotalSum: " << sTree.get_value() << std::endl;
	mTree.push(maxPe);
	sTree.push(maxPe);
}

at::Tensor PESampler::sample(int64_t batch_size) {
	at::Tensor indexes = at::empty(
		{ batch_size },
		at::TensorOptions().dtype(c10::ScalarType::Long)
	);

	float segment = sTree.get_value() / batch_size;
	float rnd;
	for (size_t i = 0; i < batch_size; i++) {
		rnd = (float)(std::rand()) / (float)(RAND_MAX);
		rnd = segment * (rnd + (float)i);
		indexes[i] = sTree.sample(rnd);
	}
	return indexes;
}

at::Tensor PESampler::get_weights(at::Tensor indexes) {
	at::Tensor weights = sTree.get(indexes);
	weights /= sTree.get_value();
	weights = (buffer_size * weights).pow(-beta);
	weights /= weights.max();
	return weights;
}

void PESampler::update(at::Tensor indexes, at::Tensor err) {
	at::Tensor pes;
	pes = err.abs() + 1e-3;
	pes = pes.pow(alpha);
	sTree.update_batch(indexes, pes);
	mTree.update_batch(indexes, pes);

	beta += 0.00005;
	beta = beta >= 1 ? 1 : beta;
}