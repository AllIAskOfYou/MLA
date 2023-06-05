#pragma once

#include "ATen/ATen.h"
#include <stdint.h>
#include "SumTree.h"
#include "MaxTree.h"

class PESampler {
public:
	PESampler(
		int64_t buffer_size,
		float alpha, 
		float beta
	);

	void push();
	at::Tensor sample(int64_t batch_size);
	at::Tensor get_weights(at::Tensor indexes);
	void update(at::Tensor indexes, at::Tensor err);

private:
	int64_t buffer_size;
	float alpha, beta;
	SumTree sTree;
	MaxTree mTree;
};