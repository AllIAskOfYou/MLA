#include "DQN.h"
#include <iostream>
#include <chrono>

DQN::DQN(
	ReplayBuffer& rb,
	int64_t batch_size,
	torch::nn::AnyModule qNet,
	torch::nn::AnyModule qNetTarget,
	torch::optim::Optimizer& opt,
	XPA& xpa,
	float gamma,
	float delta
) :
	RLA(rb, batch_size),
	qNet(qNet),
	qNetTarget(qNetTarget),
	opt(opt),
	xpa(xpa),
	gamma(gamma),
	delta(delta)
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
	auto tstart = std::chrono::high_resolution_clock::now();
	auto t0 = std::chrono::high_resolution_clock::now();
	// sample one batch from replay buffer
	RBSample rs = rb.sample(batch_size);
	auto t1 = std::chrono::high_resolution_clock::now();
	auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
	
	t0 = std::chrono::high_resolution_clock::now();
	// zero the grads
	opt.zero_grad();
	t1 = std::chrono::high_resolution_clock::now();
	auto d2 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

	t0 = std::chrono::high_resolution_clock::now();

	std::chrono::microseconds d10, d11, d12, d13, d14, d15;
	
	// create a block where no_grad is active
	at::Tensor outQT, outQ, out, outQTA, targets, loss;
	{
		at::NoGradGuard no_grad;
		t1 = std::chrono::high_resolution_clock::now();
		d10 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

		// calculate Q'(s', A')
		t0 = std::chrono::high_resolution_clock::now();
		outQT = qNetTarget.forward(rs.nStates);
		t1 = std::chrono::high_resolution_clock::now();
		d11 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

		// claculate Q(s', A')
		t0 = std::chrono::high_resolution_clock::now();
		qNet.ptr()->eval();
		outQ = qNet.forward(rs.nStates);
		t1 = std::chrono::high_resolution_clock::now();
		d12 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

		t0 = std::chrono::high_resolution_clock::now();
		// calculate Q(s', a')
		outQTA = outQT.index({ at::arange(batch_size), outQ.argmax(1) });
		t1 = std::chrono::high_resolution_clock::now();
		d13 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

		// calculate target Q(s, a)
		t0 = std::chrono::high_resolution_clock::now();
		targets = rs.rewards + gamma * outQTA;
		t1 = std::chrono::high_resolution_clock::now();
		d14 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
		//std::cout << "rewards: \n" << rs.rewards << std::endl;
		//std::cout << "outQ: \n" << outQ << std::endl;
		//std::cout << "outQT: \n" << outQT << std::endl;

		// detach from graph for faster computation. we don't want to update targets parameters
		// not really needed since NoGradAuard is active, but just to be save
		t0 = std::chrono::high_resolution_clock::now();
		targets = targets.detach();
		t1 = std::chrono::high_resolution_clock::now();
		d15 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
		t0 = std::chrono::high_resolution_clock::now();
	}
	
	t1 = std::chrono::high_resolution_clock::now();
	auto d3 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
	// calculate Q(s, A)
	qNet.ptr()->train();
	out = qNet.forward(rs.states);

	// calculate Q(s, a)
	out = out.index({ at::arange(batch_size), rs.aActions });

	t0 = std::chrono::high_resolution_clock::now();
	loss = torch::nn::MSELoss()(out, targets);//(outQ - targets).square().mean();
	t1 = std::chrono::high_resolution_clock::now();
	auto d16 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

	t0 = std::chrono::high_resolution_clock::now();
	loss.backward();
	t1 = std::chrono::high_resolution_clock::now();
	auto d4 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
	t0 = std::chrono::high_resolution_clock::now();
	opt.step();
	t1 = std::chrono::high_resolution_clock::now();
	auto d5 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);


	// update target net parameters to be more like online net
	t0 = std::chrono::high_resolution_clock::now();
	update_params(delta);
	t1 = std::chrono::high_resolution_clock::now();
	auto d6 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

	auto tend = std::chrono::high_resolution_clock::now();
	auto dfull = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart);

	/*
	std::cout << "sample time: \n" << d1.count() << std::endl;
	std::cout << "zero grads time: \n" << d2.count() << std::endl;
	std::cout << "get targets time: \n" << d3.count() << std::endl;
	std::cout << "loss time: \n" << d16.count() << std::endl;
	std::cout << "backward time: \n" << d4.count() << std::endl;
	std::cout << "step time: \n" << d5.count() << std::endl;
	std::cout << "update params time: \n" << d6.count() << std::endl;
	std::cout << "no grad time: \n" << d10.count() << std::endl;
	std::cout << "first call time: \n" << d11.count() << std::endl;
	std::cout << "second call time: \n" << d12.count() << std::endl;
	std::cout << "index time: \n" << d13.count() << std::endl;
	std::cout << "plus time: \n" << d14.count() << std::endl;
	std::cout << "detach time: \n" << d15.count() << std::endl;
	std::cout << "update full time: \n" << dfull.count() << std::endl;
	*/
}

// calculates new action given the curent policy
at::Tensor DQN::nextAction() {
	State state = rb.get(-1);
	qNet.ptr()->eval();
	at::Tensor out =  qNet.forward(state);
	//std::cout << out << std::endl;

	at::Tensor nAction = xpa.nextAction(out);
	std::cout << nAction << std::endl;

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
	//std::cout << out << std::endl;

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