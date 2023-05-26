#pragma once

#include <ATen/ATen.h>
#include <stdint.h>
#include "XPA.h"
#include <cmath>

class EpsilonGreedy : public XPA {
public:
	EpsilonGreedy(float epsilon, float minEpsilon, float steps);

	void update();
	at::Tensor nextAction(at::Tensor qvalue);
	at::Tensor prob(at::Tensor qvalue);

private:
	float epsilon, minEpsilon, decay;
};

