#include "DQN.h"
#include <iostream>

DQN::DQN(
	ReplayBuffer& rb, 
	int64_t batch_size,
	torch::optim::Optimizer& opt,
	float gamma,
	float delta
) :
	RLA(rb, batch_size),
	opt(opt),
	gamma(gamma),
	delta(delta)
{}

void DQN::update() {
	RBSample rs = rb.sample(batch_size);
	std::cout << "dela\n" << rs.nStates << std::endl;
	at::NoGradGuard no_grad;
	no_grad;
	at::Tensor outQT = qNetTarget.forward(
		rs.states,
		rs.aActions,
		rs.oActions
	);
	at::Tensor outQ = qNet.forward(
		rs.states,
		rs.aActions,
		rs.oActions
	);
	at::Tensor targets = rs.rewards + 
		gamma * outQT.index({at::arange(batch_size), outQ.argmax(1)});
	targets = targets.detach();
	torch::nn::MSELoss outQ, targets
}

at::Tensor DQN::nextAction() {
	return at::zeros({1});
}