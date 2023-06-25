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
	torch::nn::AnyModule iCMNet,
	float lambda,
	float beta,
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
	iCMNet(iCMNet),
	lambda(lambda),
	beta(beta),
	ro(ro)
{}

void DQNA::update() {
	// sample one batch from replay buffer
	RBSample rs = rb.sample(batch_size);

	at::Tensor err, a_pred, loss, loss_inv, loss_fw;

	// zero the grads
	opt.zero_grad();

	// calculate next oa
	ICMReturn out = iCMNet.forward<ICMReturn>(rs);

	// inverse module
	a_pred = out.aPred.softmax(1);
	a_pred = a_pred.index(
		{ at::arange(batch_size), rs.nStates.aActions.index({at::indexing::Slice(), -1}) }
	);
	loss_inv = -a_pred.log().mean();

	// forward module
	if (rb.update_steps == 10000) {
		std::cout << out.sNext << std::endl;
		std::cout << out.sNextPred << std::endl;
	}
	loss_fw = 0.5 * (out.sNext - out.sNextPred).square().sum(1).sqrt();

	// add intrinsic reward to the extrinsic reward
	if (rb.update_steps % 100 == 0) {
		std::cout << "APred: " << a_pred.detach().mean().item<float>() << std::endl;
		std::cout << "Inv-Loss: " << loss_fw.mean().item<float>() << std::endl;
	}
	rs.rewards += ro * loss_fw.detach();

	loss_fw = loss_fw.mean();

	// calculate q err
	err = calculate_err(rs);
	loss = err.square().mean();

	// join loss
	loss = lambda * loss + (1 - lambda) * ((1 - beta) * loss_inv + beta * loss_fw);
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