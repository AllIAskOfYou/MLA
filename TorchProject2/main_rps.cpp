#include "main_rps.h"
#include "GameSession.h"
#include "DQN.h"
#include "models.h"
#include "EpsilonGreedy.h"

int main_rps() {
	int64_t es_n = 1;
	int64_t as_n = 1;
	int64_t a_n = 4;
	int64_t last_n = 32;
	int64_t batch_size = 32;
	int64_t buffer_size = 128;
	int64_t max_iter = 1000;

	std::vector<int64_t> dims = { 8, 8 };
	std::vector<bool> pools = { true, false};
	std::vector<int64_t> units = { 8, a_n };


	ReplayBuffer rb(buffer_size, last_n, es_n, as_n);

	torch::nn::AnyModule qNet(
		md::QNetConv(es_n, as_n, a_n, last_n, 8, 8, 8, dims, pools, units)
	);
	torch::nn::AnyModule qNetTarget(
		md::QNetConv(es_n, as_n, a_n, last_n, 8, 8, 8, dims, pools, units)
	);

	//torch::nn::AnyModule qNet(md::QNetState(s_n, a_n, last_n));
	//torch::nn::AnyModule qNetTarget(md::QNetState(s_n, a_n, last_n));

	torch::optim::Adam opt(qNet.ptr()->parameters(), torch::optim::AdamOptions(0.001));

	EpsilonGreedy xpa(1, 0.1, 0.99);

	DQN dqn(rb, batch_size, qNet, qNetTarget, opt, xpa, 0.5, 0.99);

	GameSession gs(es_n, as_n, a_n, dqn, max_iter);
	gs.start();
}