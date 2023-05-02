#include "EpsilonGreedy.h"
#include <iostream>

EpsilonGreedy::EpsilonGreedy(
	float epsilon,
	float minEpsilon,
	float decay
) :
	epsilon(epsilon),
	minEpsilon(minEpsilon),
	decay(decay)
{}

void EpsilonGreedy::update() {
	epsilon *= decay;
	if (epsilon < minEpsilon) {
		epsilon = minEpsilon;
	}
}

at::Tensor EpsilonGreedy::nextAction(at::Tensor qvalue) {
	float rand_int = (float)(std::rand()) / (float)(RAND_MAX);
	at::Tensor nAction;
	if (rand_int > epsilon) {
		nAction = qvalue.argmax();
	}
	else {
		nAction = at::randint(0, qvalue.size(1), { 1 });
	}
	std::cout << "Epsilon: \n" << epsilon << std::endl;
	return nAction;
}