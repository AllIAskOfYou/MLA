#include "PDQN.h"

PDQN::PDQN(
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
	PESampler& pes
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
	pes(pes)
{}

void PDQN::push (
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
	pes.push();
}

void PDQN::update() {
	// sample one batch from priority replay buffer
	at::Tensor indexes = pes.sample(batch_size);
	RBSample rs = rb.get_sample(indexes);

	// calculate weights from pes
	at::Tensor weights = pes.get_weights(indexes);

	// zero the grads
	opt.zero_grad();

	at::Tensor err, pes_up, loss;
	err = calculate_err(rs);

	// update priorities of sampled batch
	pes.update(indexes, err.detach());

	// calculate loss and update parameters
	std::cout << "W: " << weights.sizes() << "ERR: " << err.sizes() << std::endl;
	loss = (weights * err.square()).sum();
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