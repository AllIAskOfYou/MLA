#pragma once

#include <ATen/ATen.h>
#include <stdint.h>
#include "XPA.h"

class EpsilonGreedy : public XPA {
public:
	EpsilonGreedy(float epsilon, float minEpsilon, float decay);

	void update();
	at::Tensor nextAction(at::Tensor qvalue);
	at::Tensor prob(at::Tensor qvalue);

private:
	float epsilon, minEpsilon, decay;
};

