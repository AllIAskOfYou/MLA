#include "PDQN.h"

PDQN::PDQN(
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
	DQN(
		rb,
		batch_size,
		qNet,
		qNetTarget,
		opt,
		xpa,
		gamma,
		delta,
		pUpdateWait
	),
	mTree(batch_size),
	sTree(batch_size)
{}

void PDQN::push(
	at::Tensor es,
	at::Tensor as,
	at::Tensor os,
	at::Tensor aa,
	at::Tensor oa,
	at::Tensor r,
	at::Tensor t
)
{
	DQN::push(es, as, os, aa, oa, r, t);
	float maxPe = mTree.get_value();
	maxPe = (maxPe == 0) ? 1 : maxPe;
	mTree.push(maxPe);
	sTree.push(maxPe);
}

void PDQN::update() {
	// sample one batch from priority replay buffer
	auto indexes = sTree.sample_batch(batch_size);
	at::Tensor pes = at::from_blob(sTree.get(indexes).data(), (int)indexes.size());
	at::Tensor idx = at::from_blob(indexes.data(), { batch_size });
	RBSample rs = rb.get_sample(idx);

	// calculate weights from pes
	auto pp = pes / sTree.get_value();
	auto weights = (rb.get_size() * pp).pow(-beta);

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
	targets = rs.rewards + gamma * outQTA * rs.terminal;

	// calculate Q(s, A)
	qNet.ptr()->train();
	out = qNet.forward(rs.states);

	//std::cout << rs.states.aStates.index({ 0 }) << " : " << rs.aActions.index({ 0 }) << " -> " << rs.rewards.index({ 0 }) << "Out: " << out.index({0}) << std::endl;

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