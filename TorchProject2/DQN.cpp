#include "DQN.h"
#include <iostream>

DQN::DQN(ReplayBuffer& rb) : RLA(rb) {
	
}

void DQN::update() {
	RBSample rs = rb.sample(4);
	std::cout << "dela\n" << rs.nStates << std::endl;
}

at::Tensor DQN::nextAction() {
	return at::zeros({1});
}