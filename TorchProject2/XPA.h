#pragma once

#include <ATen/ATen.h>

class XPA {
protected:
	XPA() {};

public:
	virtual void update() = 0;
	virtual at::Tensor nextAction(at::Tensor qvalue) = 0;
	virtual at::Tensor prob(at::Tensor qvalue) = 0;
};