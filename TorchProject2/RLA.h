#pragma once

#include <ATen/ATen.h>
#include <stdint.h>
#include "ReplayBuffer.h"

class RLA {
protected:
	RLA(ReplayBuffer& rb, int64_t batch_size) :
		rb(rb),
		batch_size(batch_size)
	{}

public:
	virtual void push(
		at::Tensor es,
		at::Tensor as,
		at::Tensor os,
		at::Tensor aa,
		at::Tensor oa,
		at::Tensor r,
		at::Tensor t
	)
	{
		rb.push(es, as, os, aa, oa, r, t);
	}

public:
	virtual void update() = 0;
	virtual at::Tensor nextAction() = 0;
	virtual at::Tensor selfPlay() = 0;

protected:
	ReplayBuffer& rb;
	int64_t batch_size;
};