#pragma once

#include "DTensor.h"
#include "ATen/ATen.h"
#include <stdint.h>

typedef struct RBSample {
	at::Tensor states;		// state			(t)
	at::Tensor aActions;	// agent action		(t)
	at::Tensor oActions;	// oponent action	(t)
	at::Tensor rewards;		// reward			(t)
	at::Tensor nStates;		// next state		(t+1)
	at::Tensor nOActions;	// next oponent a	(t+1)
} RBSample;

class ReplayBuffer {
public:
	ReplayBuffer() {};
	ReplayBuffer(int64_t size, int64_t last_n, int64_t s_n);

	void pushState(at::Tensor s);

	void push(at::Tensor s, at::Tensor aa, at::Tensor oa, at::Tensor r);

	RBSample sample(int64_t batchSize);

	RBSample get(int64_t index);

private:
	int64_t size;
	DTensor states;
	DTensor aActions;
	DTensor oActions;
	DTensor rewards;
	at::Tensor prob;
};