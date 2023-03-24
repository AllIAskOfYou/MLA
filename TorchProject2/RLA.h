#pragma once

#include <ATen/ATen.h>
#include "ReplayBuffer.h"

class RLA {
protected:
	RLA(ReplayBuffer& rb) : rb(rb) {}

public:
	virtual void update() = 0;
	virtual at::Tensor nextAction() = 0;

protected:
	ReplayBuffer& rb;
};