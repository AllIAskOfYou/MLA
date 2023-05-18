#include "DQN.h"
#include <iostream>

DQN::DQN(
	ReplayBuffer& rb,
	int64_t batch_size,
	torch::nn::AnyModule qNet,
	torch::nn::AnyModule qNetTarget,
	torch::optim::Optimizer& opt,
	XPA& xpa,
	float gamma,
	float delta,
	int pUpdateWait
) :
	RLA(rb, batch_size),
	qNet(qNet),
	qNetTarget(qNetTarget),
	opt(opt),
	xpa(xpa),
	gamma(gamma),
	delta(delta),
	pUpdateWait(pUpdateWait)
{
	// make the target network the same as the online net
	update_params(0);

	// set Q' - target net to eval mode permanently
	qNetTarget.ptr()->eval();
}


// s  - curent state
// s' - next state
// A  - all actions
// a  - specific action
// A' - all next actions
// a' - specific next action
// Q  - q value
// Q' - q value from target net

void DQN::update() {
	//auto t0 = std::chrono::high_resolution_clock::now();

	//std::thread t(&DQN::update_params, this, delta);

	// sample one batch from replay buffer
	RBSample rs = rb.sample(batch_size);
	
	// zero the grads
	opt.zero_grad();

	// create a block where no_grad is active
	at::Tensor outQT, outQ, probA, out, outQTA, targets, loss;
	{
		at::NoGradGuard no_grad;

		// calculate Q'(s', A')
		outQT = qNetTarget.forward(rs.nStates);

		// calculate Q(s', A')
		qNet.ptr()->eval();
		outQ = qNet.forward(rs.nStates);
		
		// detach from graph for faster computation. we don't want to update targets parameters
		// not really needed since NoGradAuard is active, but just to be save
		outQT = outQT.detach();
		outQ = outQ.detach();
	}

	// calculate prob(A' | s')
	probA = xpa.prob(outQ);

	// calculate E(Q(s', A')) = SUM( prob(a'_i | s') * outQT(s', a'_i)) 
	outQTA = (probA * outQT).sum(1);

	// calculate Q(s', a')
	// auto outQTA2 = outQT.index({ at::arange(batch_size), outQ.argmax(1) });

	// calculate target Q(s, a)
	targets = rs.rewards + gamma * outQTA;
	
	// calculate Q(s, A)
	qNet.ptr()->train();
	out = qNet.forward(rs.states);

	std::cout << rs.states.aStates.index({ 0 }) << " : " << rs.aActions.index({ 0 }) << " -> " << rs.rewards.index({ 0 }) << "Out: " << out.index({0}) << std::endl;

	// calculate Q(s, a)
	out = out.index({ at::arange(batch_size), rs.aActions });

	loss = torch::nn::MSELoss()(out, targets);//(outQ - targets).square().mean();
	loss.backward();
	opt.step();

	// update target net parameters to be more like online net
	if (pUpdateTimes == pUpdateWait) {
		update_params(delta);
		pUpdateTimes = 0;
	}
	else {
		pUpdateTimes++;
	}
	
}

// calculates new action given the curent policy
at::Tensor DQN::nextAction() {
	State state = rb.get(-1);

	qNet.ptr()->eval();
	at::Tensor out =  qNet.forward(state);

	at::Tensor nAction = xpa.nextAction(out);

	// update exploration method
	xpa.update();

	return nAction;
}

// calculates new action for self oponent given the older q-net policy
at::Tensor DQN::selfPlay() {
	State state = rb.get(-1);
	
	// inverse the last state to change the perspective from agent to opponent
	state = state.inverse();

	// predict on the older target q-net
	at::Tensor out = qNetTarget.forward(state);

	at::Tensor nAction = xpa.nextAction(out);

	return nAction;
}

void DQN::update_params(float delta) {
	//std::cout << "Delta: " << delta << std::endl;
	auto p = qNet.ptr()->parameters();
	auto pT = qNetTarget.ptr()->parameters();
	const size_t p_n = pT.size();
	for (size_t i = 0; i < p_n; i++) {
		pT[i].set_data(delta * pT[i] + (1 - delta) * p[i]);
	}
	auto b = qNet.ptr()->buffers();
	auto bT = qNetTarget.ptr()->buffers();
	const size_t b_n = bT.size();
	for (size_t i = 0; i < b_n; i++) {
		bT[i].set_data(delta * bT[i] + (1 - delta) * b[i]);
	}

}