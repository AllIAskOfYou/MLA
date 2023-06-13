#include "DQNA.h"

DQNA::DQNA(
	ReplayBuffer& rb,
	int64_t batch_size,
	torch::nn::AnyModule qNet,
	torch::nn::AnyModule qNetTarget,
	torch::optim::Optimizer& opt,
	torch::optim::LRScheduler& lrs,
	XPA& xpa,
	float gamma,
	float delta,
	int pUpdateWait,
	torch::nn::AnyModule oaNet,
	//torch::optim::Optimizer& oaNetOpt,
	//torch::optim::LRScheduler& oaNetLrs
	float lambda,
	float ro
) :
	DQN(
		rb,
		batch_size,
		qNet,
		qNetTarget,
		opt,
		lrs,
		xpa,
		gamma,
		delta,
		pUpdateWait
	),
	oaNet(oaNet),
	//oaNetOpt(oaNetOpt),
	//oaNetLrs(oaNetLrs),
	lambda(lambda),
	ro(ro)
{}

void DQNA::update() {
	// sample one batch from replay buffer
	RBSample rs = rb.sample(batch_size);

	at::Tensor err, loss, target_oa, target_oa_next, oa_loss, rewards;

	// zero the grads
	opt.zero_grad();
	//oaNetOpt.zero_grad();

	// calculate next oa
	target_oa = oaNet.forward(rs.states).softmax(1);
	target_oa = target_oa.index({ at::arange(batch_size), rs.nStates.oActions.flatten() });

	//target_oa_next = oaNet.forward(rs.nStates);

	// add intrinsic reward to the extrinsic reward
	//std::cout << "Intrinsic: " << target_oa.detach()[0] << " : " << rs.nStates.oActions.flatten()[0] << std::endl;
	rs.rewards += ro * (1 - target_oa.detach());

	// calculate action pred loss
	oa_loss = -target_oa.log().mean();// torch::nn::CrossEntropyLoss()(target_oa, rs.nStates.oActions);

	// calculate q err
	err = calculate_err(rs);
	loss = err.square().mean();

	// join loss
	loss = lambda * loss + oa_loss;
	loss.backward();
	opt.step();
	lrs.step();

	// update target net parameters to be more like online net
	if (pUpdateTimes == pUpdateWait) {
		update_params(delta);
		pUpdateTimes = 0;
	}
	else {
		pUpdateTimes++;
	}
}