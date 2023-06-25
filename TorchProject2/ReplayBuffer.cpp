#include "ReplayBuffer.h"
#include <iostream>

ReplayBuffer::ReplayBuffer(int64_t size, int64_t last_n, int64_t es_n, int64_t as_n ) :
	size(size)
{
	eStates = DTensor({ size + 1, last_n, es_n }, at::TensorOptions().dtype(c10::ScalarType::Float));
	aStates = DTensor({ size + 1, last_n, as_n }, at::TensorOptions().dtype(c10::ScalarType::Float));
	oStates = DTensor({ size + 1, last_n, as_n }, at::TensorOptions().dtype(c10::ScalarType::Float));
	aActionsOut = DTensor({ size + 1, 1 }, at::TensorOptions().dtype(c10::ScalarType::Long));
	aActions = DTensor({ size + 1, last_n }, at::TensorOptions().dtype(c10::ScalarType::Long));
	oActions = DTensor({ size + 1, last_n }, at::TensorOptions().dtype(c10::ScalarType::Long));
	rewards = DTensor({ size + 1, 1 }, at::TensorOptions().dtype(c10::ScalarType::Float));
	terminal = DTensor({ size + 1, 1 }, at::TensorOptions().dtype(c10::ScalarType::Long));
	prob = at::full({ size }, 1.0 / size, at::TensorOptions().dtype(c10::ScalarType::Float));
}

void ReplayBuffer::push(
	at::Tensor es, at::Tensor as, at::Tensor os,
	at::Tensor aa_out, at::Tensor aa, at::Tensor oa,at::Tensor r,
	at::Tensor t)
{
	if (update_steps % 1000 == 0) {
		std::cout << "Step: " << update_steps << std::endl;
	}
	eStates.push(es);
	aStates.push(as);
	oStates.push(os);
	aActionsOut.push(aa_out);
	aActions.push(aa);
	oActions.push(oa);
	rewards.push(r);
	terminal.push(t);

	update_steps++;
}

void ReplayBuffer::pushEmpty() {
	eStates.pushEmpty();
	aStates.pushEmpty();
	oStates.pushEmpty();
	aActionsOut.pushEmpty();
	aActions.pushEmpty();
	oActions.pushEmpty();
	rewards.pushEmpty();
	terminal.pushEmpty();
}

RBSample ReplayBuffer::get_sample(at::Tensor idx) {
	RBSample smpl;
	smpl.states = get(idx);
	smpl.aActions = aActionsOut.index(idx + 1).flatten();
	smpl.rewards = rewards.index(idx + 1).flatten();
	smpl.nStates = get(idx + 1);
	smpl.terminal = terminal.index(idx + 1).flatten();
	return smpl;
}

RBSample ReplayBuffer::sample(int64_t batchSize) {
	at::Tensor idx = prob.multinomial(batchSize, false);
	return get_sample(idx);
}


State ReplayBuffer::get(int64_t index) {
	at::Tensor idx = at::tensor({index});
	return get(idx);	
}

State ReplayBuffer::get(at::Tensor idx) {
	State state;
	state.eStates = eStates.index(idx);
	state.aStates = aStates.index(idx);
	state.oStates = oStates.index(idx);

	state.aActions = aActions.index(idx);
	state.oActions = oActions.index(idx);

	return state;
}