#pragma once

#include "XPA.h"

class Boltzmann : public XPA {
public:
	Boltzmann(float t, float minT, float decay);

	void update();
	at::Tensor nextAction(at::Tensor qvalue);

	at::Tensor prob(at::Tensor qvalue);

private:
	float t, minT, decay;
	float err = 1e-2;
};