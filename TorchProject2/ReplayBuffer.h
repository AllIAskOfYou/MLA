#pragma once

#include "ATen/ATen.h"
#include <stdint.h>
#include "DTensor.h"
#include "RBSample.h"

// Context:
//
// { s(t), aa(t), oa(t) } -> { s(t+1), r(t) }
// Q( s(t), aa(t-1), oa(t-1) ) -> Q_A

class ReplayBuffer {
public:
	ReplayBuffer() {};
	ReplayBuffer(int64_t size, int64_t last_n, int64_t es_n, int64_t as_n);

	void push(
		at::Tensor es, at::Tensor as, at::Tensor os,
		at::Tensor aa, at::Tensor oa, at::Tensor r
	);

	RBSample sample(int64_t batchSize);

	State get(at::Tensor idx);
	State get(int64_t index);

private:
	int64_t size;
	DTensor eStates;
	DTensor aStates;
	DTensor oStates;
	DTensor aActions;
	DTensor oActions;
	DTensor rewards;
	at::Tensor prob;

public:
	int64_t update_steps = 0;
};