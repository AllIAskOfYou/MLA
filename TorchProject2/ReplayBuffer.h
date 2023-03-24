#pragma once

#include "DTensor.h"
#include "ATen/ATen.h"
#include <stdint.h>

typedef struct RBSample {
	at::Tensor states;
	at::Tensor aActions;
	at::Tensor oActions;
	at::Tensor rewards;
	at::Tensor nStates;
} RBSample;

class ReplayBuffer {
public:
	ReplayBuffer() {};
	ReplayBuffer(int64_t size, int64_t last_n, int64_t s_n);

	void pushState(at::Tensor s);

	void push(at::Tensor s, at::Tensor aa, at::Tensor oa, at::Tensor r);

	RBSample sample(int64_t batchSize);

private:
	int64_t size;
	DTensor states;
	DTensor aActions;
	DTensor oActions;
	DTensor rewards;
	at::Tensor prob;
};