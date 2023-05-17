#include "ReplayBuffer.h"
#include <iostream>

ReplayBuffer::ReplayBuffer(int64_t size, int64_t last_n, int64_t es_n, int64_t as_n ) :
	size(size)
{
	eStates = DTensor({ size + 1, last_n, es_n }, at::TensorOptions().dtype(c10::ScalarType::Float));
	aStates = DTensor({ size + 1, last_n, as_n }, at::TensorOptions().dtype(c10::ScalarType::Float));
	oStates = DTensor({ size + 1, last_n, as_n }, at::TensorOptions().dtype(c10::ScalarType::Float));
	aActions = DTensor({ size + 1, last_n }, at::TensorOptions().dtype(c10::ScalarType::Long));
	oActions = DTensor({ size + 1, last_n }, at::TensorOptions().dtype(c10::ScalarType::Long));
	rewards = DTensor({ size + 1, 1 }, at::TensorOptions().dtype(c10::ScalarType::Float));
	prob = at::full({ size }, 1.0 / size, at::TensorOptions().dtype(c10::ScalarType::Float));
}

void ReplayBuffer::push(
	at::Tensor es, at::Tensor as, at::Tensor os,
	at::Tensor aa, at::Tensor oa,at::Tensor r)
{
	eStates.push(es);
	aStates.push(as);
	oStates.push(os);
	aActions.push(aa);
	oActions.push(oa);
	rewards.push(r);
}

RBSample ReplayBuffer::sample(int64_t batchSize) {
	at::Tensor idx = prob.multinomial(batchSize, false) + 1;
	RBSample smpl;

	//std::cout << aStates.index(at::arange(size+1)) << std::endl;
	smpl.states = get(idx - 1);
	smpl.aActions = aActions.index(idx).index({at::indexing::Slice(), -1});
	smpl.rewards = rewards.index(idx).flatten();
	smpl.nStates = get(idx);

	return smpl;
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