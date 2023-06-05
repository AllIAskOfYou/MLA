#pragma once

#include "Tree.h"

class SumTree : public Tree {
public:
	SumTree(int64_t size);

	void update_(int64_t index, float pe);
	int64_t sample(float pe);
	at::Tensor sample_batch(int64_t batch_size);
};