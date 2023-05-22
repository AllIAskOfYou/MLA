#include "main_rpg.h"

#include "GameSession.h"
#include "DQN.h"
#include "models.h"
#include "EpsilonGreedy.h"
#include "Boltzmann.h"

int main_rpg() {
	int64_t es_n = 1;
	int64_t as_n = 2;
	int64_t a_n = 5;
	int64_t last_n = 1;						// 1 -> 16,		Update time: 16 ms -> 32 ms		TURBO: 24 ms
	int64_t batch_size = 256;				// 128 -> 512,	Update time: 14.7 ms -> 16.4 ms
	int64_t buffer_size = 512;				// Didn't change from 128 -> 1024
	int64_t max_iter = 1000; // legacy

	std::vector<int64_t> dims = { 4 };
	std::vector<bool> pools = { false };
	std::vector<int64_t> units = { 4, a_n };


	ReplayBuffer rb(buffer_size, last_n, es_n, as_n);

	torch::nn::AnyModule qNet(
		md::QNetConv(es_n, as_n, a_n, last_n, 4, 4, 4, dims, pools, units)
	);
	torch::nn::AnyModule qNetTarget(
		md::QNetConv(es_n, as_n, a_n, last_n, 4, 4, 4, dims, pools, units)
	);

	//torch::nn::AnyModule qNet(md::QNetState(s_n, a_n, last_n));
	//torch::nn::AnyModule qNetTarget(md::QNetState(s_n, a_n, last_n));

	torch::optim::Adam opt(qNet.ptr()->parameters(), torch::optim::AdamOptions(0.01));

	EpsilonGreedy xpa(2.2, 0.25, 0.994);
	//Boltzmann xpa(10, 0.5, 0.99);

	DQN dqn(rb, batch_size, qNet, qNetTarget, opt, xpa, 0, 0.98, 0);

	GameSession gs(es_n, as_n, a_n, dqn, max_iter);
	gs.start();
	return 0;
}