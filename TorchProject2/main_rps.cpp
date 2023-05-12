#include "main_rps.h"
#include "GameSession.h"
#include "DQN.h"
#include "models.h"
#include "EpsilonGreedy.h"

int main_rps() {
	int64_t es_n = 1;
	int64_t as_n = 1;
	int64_t a_n = 4;
	int64_t last_n = 4;
	int64_t batch_size = 8;
	int64_t buffer_size = 16;
	int64_t max_iter = 1000;

	std::vector<int64_t> dims = { 8, 8 };
	std::vector<bool> pools = { true, false};
	std::vector<int64_t> units = { 8, a_n };


	ReplayBuffer rb(buffer_size, last_n, es_n, as_n);

	torch::nn::AnyModule qNet(
		md::QNetConv(es_n, as_n, a_n, last_n, 8, 8, 8, dims, pools, units)
	);
	//auto net = md::ConvBlock(3, 12, true);
	//std::cout << "before" << std::endl;
	//md::ConvBlockImpl& net;
	//torch::nn::AnyModule net(md::QNetConv(qNet.ptr()->clone()));
	//std::cout << "after" << std::endl;

	at::Tensor a = at::tensor({ 1,2,3,4 }, at::TensorOptions().dtype(at::kDouble)).view({ 2,2 });
	at::Tensor b = at::tensor({ 5,6,7,8 }, at::TensorOptions().dtype(at::kFloat)).view({ 2,2 });
	b[0, 0] = 3.12315123;
	auto c = a;
	c.copy_(b);
	std::cout << a << b << c << std::endl;


	torch::nn::AnyModule qNetTarget(
		md::QNetConv(es_n, as_n, a_n, last_n, 8, 8, 8, dims, pools, units)
	);

	//torch::nn::AnyModule qNet(md::QNetState(s_n, a_n, last_n));
	//torch::nn::AnyModule qNetTarget(md::QNetState(s_n, a_n, last_n));

	torch::optim::Adam opt(qNet.ptr()->parameters(), torch::optim::AdamOptions(0.001));

	EpsilonGreedy xpa(1, 0.1, 0.8);

	DQN dqn(rb, batch_size, qNet, qNetTarget, opt, xpa, 0.5, 0);

	GameSession gs(es_n, as_n, a_n, dqn, max_iter);
	gs.start();
}