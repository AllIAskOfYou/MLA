#include "EpsilonGreedy.h"
#include <iostream>

EpsilonGreedy::EpsilonGreedy(
	float epsilon,
	float minEpsilon,
	float steps
) :
	epsilon(epsilon),
	minEpsilon(minEpsilon),
	decay(pow(minEpsilon/epsilon, 1/steps))
{}

void EpsilonGreedy::update() {
	epsilon *= decay;
	if (epsilon < minEpsilon) {
		epsilon = minEpsilon;
	}
	else {
		std::cout << "Epsilon: \n" << epsilon << std::endl;
	}
}

at::Tensor EpsilonGreedy::nextAction(at::Tensor qvalue) {
	float rand_int = (float)(std::rand()) / (float)(RAND_MAX);
	at::Tensor nAction;
	if (rand_int > epsilon) {
		nAction = qvalue.argmax(1);
	}
	else {
		nAction = at::randint(0, qvalue.size(1), { 1 });
	}
	return nAction;
}

at::Tensor EpsilonGreedy::prob(at::Tensor qvalue) {
	at::Tensor prob = at::zeros(qvalue.sizes(), at::TensorOptions().dtype(at::kFloat));
	prob.index_put_({at::arange(qvalue.size(0)), qvalue.argmax(1)}, 1);
	return prob;
}