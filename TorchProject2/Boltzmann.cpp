#include "Boltzmann.h"
#include <iostream>

Boltzmann::Boltzmann(
	float t,
	float minT,
	float decay
) :
	t(t),
	minT(minT),
	decay(decay)
{}

void Boltzmann::update() {
	t *= decay;
	if (t < minT) {
		t = minT;
	}
}

at::Tensor Boltzmann::nextAction(at::Tensor qvalue) {
	//std::cout << "Temp: " << t << std::endl;

	//std::cout << "qvalue: " << qvalue << std::endl;
	at::Tensor aAction;
	aAction = prob(qvalue);
	//std::cout << "Softmx: " << aAction << std::endl;
	aAction = aAction.multinomial(1, false).view({1});
	//std::cout << "Sample: " << aAction << std::endl;
	return aAction;
}

at::Tensor Boltzmann::prob(at::Tensor qvalue) {
	return at::softmax(qvalue * (1 / (t + err)), 1);
}