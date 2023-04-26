#include "ReplayBuffer.h"

ReplayBuffer::ReplayBuffer(int64_t size, int64_t last_n, int64_t s_n ) :
	size(size)
{
	states = DTensor({ size + 1, last_n, s_n }, at::TensorOptions().dtype(c10::ScalarType::Float));
	aActions = DTensor({ size + 1, last_n }, at::TensorOptions().dtype(c10::ScalarType::Long));
	oActions = DTensor({ size + 1, last_n }, at::TensorOptions().dtype(c10::ScalarType::Long));
	rewards = DTensor({ size + 1, 1 }, at::TensorOptions().dtype(c10::ScalarType::Float));
	prob = at::full({ size }, 1.0 / size, at::TensorOptions().dtype(c10::ScalarType::Float));
}

void ReplayBuffer::pushState(at::Tensor s) {
	states.push(s);
}

void ReplayBuffer::push(at::Tensor s, at::Tensor aa, at::Tensor oa, at::Tensor r) {
	states.push(s);
	aActions.push(aa);
	oActions.push(oa);
	rewards.push(r);
}

RBSample ReplayBuffer::sample(int64_t batchSize) {
	at::Tensor idx = prob.multinomial(batchSize, false) + 1;
	RBSample smpl;
	smpl.states = states.index(idx - 1);
	smpl.aActions = aActions.index(idx);
	smpl.oActions = oActions.index(idx);
	smpl.rewards = rewards.index(idx).flatten();
	smpl.nStates = states.index(idx);
	smpl.pAActions = states.index(idx - 1);
	smpl.pOActions = oActions.index(idx - 1);

	return smpl;
}

// only sets state, aaction and oaction for now !!
RBSample ReplayBuffer::get(int64_t index) {
	at::Tensor idx = at::tensor({index});
	RBSample smpl;
	smpl.states = states.index(idx);
	smpl.pAActions = states.index(idx);
	smpl.pOActions = oActions.index(idx);
	
	return smpl;
}