#include "main_rpg.h"

#include "GameSession.h"
#include "DQN.h"
#include "models.h"
#include "EpsilonGreedy.h"
#include "Boltzmann.h"

int main_rpg() {
	int64_t es_n = 1;
	int64_t as_n = 1;
	int64_t a_n = 5;
	int64_t last_n = 1;
	int64_t batch_size = 64;
	int64_t buffer_size = 128;
	int64_t max_iter = 1000; // legacy

	std::vector<int64_t> dims = { 4 };
	std::vector<bool> pools = { false };
	std::vector<int64_t> units = { 4, a_n };


	ReplayBuffer rb(buffer_size, last_n, es_n, as_n);

	/*
	torch::nn::Linear l(2, 3);

	torch::nn::Linear r = l;

	auto lp = l->parameters();
	std::cout << "params: \n" << lp[0] << std::endl;
	auto rp = r->parameters();
	rp[0].set_data(rp[0] + 2);
	std::cout << "params: \n" << lp[0] << std::endl;
	*/

	torch::nn::AnyModule qNet(
		md::QNetConv(es_n, as_n, a_n, last_n, 4, 4, 4, dims, pools, units)
	);
	torch::nn::AnyModule qNetTarget(
		md::QNetConv(es_n, as_n, a_n, last_n, 4, 4, 4, dims, pools, units)
	);

	//torch::nn::AnyModule qNet(md::QNetState(s_n, a_n, last_n));
	//torch::nn::AnyModule qNetTarget(md::QNetState(s_n, a_n, last_n));

	torch::optim::Adam opt(qNet.ptr()->parameters(), torch::optim::AdamOptions(0.001));

	EpsilonGreedy xpa(1, 0, 0.993);
	//Boltzmann xpa(10, 0.5, 0.99);

	DQN dqn(rb, batch_size, qNet, qNetTarget, opt, xpa, 0, 0.98);

	GameSession gs(es_n, as_n, a_n, dqn, max_iter);
	gs.start();
	return 0;
}