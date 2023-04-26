#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "DQN.h"

class QNet : public torch::nn::Module {
public:
	QNet(int64_t s_n, int64_t a_n, int64_t last_n) :
		lin(torch::nn::Linear(s_n * last_n, 8)),
		emb(torch::nn::Embedding(a_n, 8)),
		lin_emb(torch::nn::Linear(last_n * 8, 8)),
		out(torch::nn::Linear(16, a_n))
	{
		register_module("lin", lin);
		register_module("emb", emb);
		register_module("lin_emb", lin_emb);
	}

	at::Tensor forward(at::Tensor s, at::Tensor oa) {
		//s = s.transpose(1, 2);
		/*
		std::cout << s.sizes() << std::endl;
		std::cout << oa.sizes() << std::endl;
		std::cout << s.flatten(1).sizes() << std::endl;
		std::cout << oa.flatten(1).sizes() << std::endl;
		std::cout << lin(s.flatten(1)) << std::endl;
		std::cout << emb(oa.flatten(1)) << std::endl;
		*/
		return out(flatten(torch::cat({ lin(s.flatten(1)), lin_emb(emb(oa.flatten(1)).flatten(1))}, -1)));
	}

	torch::nn::Linear lin;
	torch::nn::Embedding emb;
	torch::nn::Linear lin_emb;
	torch::nn::Linear out;
	torch::nn::Flatten flatten;
};

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

	torch::nn::AnyModule qNet(QNet(s_n, a_n, last_n));
	torch::nn::AnyModule qNetTarget(QNet(s_n, a_n, last_n));

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
	std::cout << qNetTarget.forward(rs.states, rs.oActions) << std::endl;

	std::cout << "Yay" << std::endl;

	dqn.update();

	std::cout << "Update dela" << std::endl;

	at::Tensor nAction = dqn.nextAction();
	std::cout << "Next action: \n" << nAction << std::endl;
}