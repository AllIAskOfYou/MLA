#include <iostream>
#include <torch/torch.h>
#include "GameSession.h"
#include "DTensor.h"
#include "ReplayBuffer.h"
#include "DQN.h"

/*
struct RegressionImpl : torch::nn::Module {
	RegressionImpl(int s_dim, int a_dim, int last_n, int out_dim)
		: lin(torch::nn::LinearOptions(last_n * (s_dim + 2 * a_dim), out_dim))
	{
		register_module("lin", lin);
	}

	torch::Tensor forward(torch::Tensor s, torch::Tensor aa, torch::Tensor oa) {
		x = torch::cat({ flatten(s), flatten(aa), flatten(oa) }, -1);
		x = lin(x);
		return x;
	}

	torch::nn::Linear lin;
	torch::nn::Flatten flatten;
	torch::Tensor x;
};
TORCH_MODULE(Regression);
*/


int main() {
	std::cout << "Heyioo!!" << std::endl;
	int64_t s_n = 1;
	int64_t a_n = 4;
	int64_t last_n = 1;
	int64_t batch_size = 16;
	
	DTensor buf({ 3, 2, 3 }, torch::TensorOptions().dtype(torch::kFloat32));
	buf.push(torch::tensor({ 1, 1, 1 }));
	buf.push(torch::tensor({ 1, 2, 3 }));
	std::cout << buf.index(torch::tensor({0, 2, 1})) << std::endl;

	buf = DTensor({ 3, 2 }, torch::TensorOptions().dtype(torch::kInt64));
	buf.push(torch::tensor({ 1 }));
	buf.push(torch::tensor({ 2 }));
	std::cout << buf.index(torch::tensor({ 0, 2, 1 })) << std::endl;

	ReplayBuffer rb(4, 2, 3);
	at::Tensor s = at::tensor({ 7, 5, 3 }, torch::TensorOptions().dtype(torch::kFloat32));
	at::Tensor a = at::tensor({ 1 }, torch::TensorOptions().dtype(torch::kInt64));
	at::Tensor r = at::tensor({ 1 }, torch::TensorOptions().dtype(torch::kFloat32));
	
	rb.push(s, a, a, r);
	RBSample rs = rb.sample(4);
	std::cout
		<< rs.states << "\n"
		<< rs.nStates << "\n"
		<< rs.aActions << "\n"
		<< rs.oActions << "\n"
		<< rs.rewards << "\n"
		<< std::endl;

	DQN dqn = DQN(rb);
	dqn.update();
	std::shared_ptr<RLA> rla(new DQN(rb));

	rb.push(s + 3, a, a + 1, r + 1);
	rla->update();

	RLA& rl = DQN(rb);
	rl.update();





	
	/*

	Regression oaModule(s_n, a_n, last_n, a_n);
	torch::nn::AnyModule oaModel(oaModule);

	Regression envModule(s_n, a_n, last_n, s_n);
	torch::nn::AnyModule envModel(envModule);

	std::cout << oaModule << std::endl;

	std::shared_ptr<torch::optim::Optimizer> oaModelOptimizer(
		new torch::optim::SGD(oaModule->parameters(), torch::optim::SGDOptions(1))
	);

	std::shared_ptr<torch::optim::Optimizer> envModelOptimizer(
		new torch::optim::SGD(envModule->parameters(), torch::optim::SGDOptions(0.01))
	);

	//torch::optim::Optimizer* oaModelOptimizer = &optimizer

	GameSession gs(
		s_n, a_n, batch_size, last_n,
		oaModel, envModel,
		oaModelOptimizer, envModelOptimizer
	);
	gs.start();
	*/
}
