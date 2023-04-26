#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "DQN.h"
#include "models.h"



int main() {
	int64_t s_n = 5;
	int64_t a_n = 3;
	int64_t last_n = 2;
	int64_t batch_size = 4;
	int64_t buffer_size = 8;

	at::Tensor s = at::tensor({ 0, 0, 1, 2, -1 }, torch::dtype(torch::kFloat32));
	at::Tensor aa = at::tensor({ 1 }, torch::dtype(torch::kInt64));
	at::Tensor oa = at::tensor({ 2 }, torch::dtype(torch::kInt64));
	at::Tensor r = at::tensor({ 0 }, torch::dtype(torch::kFloat32));

	ReplayBuffer rb(buffer_size, last_n, s_n);
	rb.push(s, aa, oa, r);
	rb.push(s, aa, oa, r+1);
	rb.push(s, aa, oa, r+2);
	rb.push(s, aa, oa, r+3);

	torch::nn::AnyModule qNet(md::QNetState(s_n, a_n, last_n));
	torch::nn::AnyModule qNetTarget(md::QNetState(s_n, a_n, last_n));

	torch::optim::Adam opt(qNet.ptr()->parameters(), torch::optim::AdamOptions(0.001));

	std::cout << "Yay" << std::endl;

	DQN dqn(rb, batch_size, qNet, qNetTarget, opt, 0.9, 0.99);
	std::cout << "Yay" << std::endl;
	
	auto rs = dqn.get_rb().sample(buffer_size);
	std::cout << "Yay" << std::endl;
	/*
	std::cout << rs.states << "\n" << rs.oActions << std::endl;
	auto emb = torch::nn::Embedding(3, 8)(rs.oActions);
	std::cout << emb << std::endl;

	auto lin = torch::nn::Linear(3, 8)(rs.states);
	std::cout << lin << std::endl;

	auto cat = torch::cat({lin, emb}, -1);
	std::cout << cat << std::endl;
	*/
	std::cout << qNetTarget.forward(rs.states, rs.pAActions, rs.pOActions) << std::endl;

	std::cout << "Yay" << std::endl;

	at::Tensor nAction;
	nAction = dqn.nextAction();

	for (int i = 0; i < 100; i++) {
		dqn.update();
	}
	nAction = dqn.nextAction();

	for (int i = 0; i < 100; i++) {
		dqn.update();
	}
	nAction = dqn.nextAction();

	for (int i = 0; i < 100; i++) {
		dqn.update();
	}
	nAction = dqn.nextAction();

	std::cout << "Update dela" << std::endl;

	std::cout << "Next action: \n" << nAction << std::endl;
}