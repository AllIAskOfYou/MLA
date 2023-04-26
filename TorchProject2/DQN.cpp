#include "DQN.h"
#include <iostream>

DQN::DQN(
	ReplayBuffer& rb, 
	int64_t batch_size,
	torch::nn::AnyModule qNet,
	torch::nn::AnyModule qNetTarget,
	torch::optim::Optimizer& opt,
	float gamma,
	float delta
) :
	RLA(rb, batch_size),
	qNet(qNet),
	qNetTarget(qNetTarget),
	opt(opt),
	gamma(gamma),
	delta(delta)
{
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
	// sample one batch from replay buffer
	RBSample rs = rb.sample(batch_size);
	
	// zero the grads
	opt.zero_grad();
	
	// create a block where no_grad is active
	at::Tensor outQT, outQ, out, targets, loss, aActionsLast;
	{
		at::NoGradGuard no_grad;

		// calculate Q'(s', A')
		outQT = qNetTarget.forward(
			rs.nStates,
			rs.aActions,
			rs.oActions
		);

		// claculate Q(s', A')
		qNet.ptr()->eval();
		outQ = qNet.forward(
			rs.nStates,
			rs.aActions,
			rs.oActions
		);
		qNet.ptr()->train();

		// calculate target Q(s, a)
		targets = rs.rewards +
			gamma * outQT.index({ at::arange(batch_size), outQ.argmax(1) });

		// detach from graph for faster computation. we don't want to update targets parameters
		// not really needed since NoGradAuard is active, but just to be save
		targets = targets.detach();
	}
	// calculate Q(s, A)
	out = qNet.forward(
		rs.states,
		rs.pAActions,
		rs.pOActions
	);

	// calculate Q(s, a)
	aActionsLast = rs.aActions.index({ at::indexing::Slice(), -1 });
	out = out.index({ at::arange(batch_size), aActionsLast });

	loss = torch::nn::MSELoss()(out, targets);//(outQ - targets).square().mean();
	loss.backward();
	opt.step();

	// update target net parameters to be more like online net
	auto pT = qNetTarget.ptr()->parameters();
	auto p = qNet.ptr()->parameters();
	size_t p_n = pT.size();
	for (size_t i = 0; i < p_n; i++) {
		pT[i].data().copy_(delta * pT[i].data() + (1 - delta) * p[i].data());
	}
}

at::Tensor DQN::nextAction() {
	RBSample rs = rb.get(-1);
	at::Tensor out =  qNet.forward(
		rs.states,
		rs.pAActions,
		rs.pOActions
	);
	std::cout << out << std::endl;
	at::Tensor nAction = out.argmax();
	return nAction;
}