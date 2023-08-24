#include "main_rpg.h"

#include "GameSession.h"
#include "DQN.h"
#include "PDQN.h"
#include "DQNA.h"
#include "models.h"
#include "EpsilonGreedy.h"
#include "Boltzmann.h"
#include "LinearLRS.h"

int main_rpg() {
	int64_t es_n = 1;
	int64_t as_n = 7;
	int64_t a_n = 6;
	int64_t last_n = 1;						// 1 -> 16,		Update time: 16 ms -> 32 ms		TURBO: 24 ms
	int64_t batch_size = 128;				// 128 -> 512,	Update time: 14.7 ms -> 16.4 ms
	int64_t buffer_size = 10000;				// Didn't change from 128 -> 1024
	int64_t max_iter = 1000; // legacy

	std::vector<int64_t> dims = { 16 };
	std::vector<bool> pools = { false };
	//std::vector<int64_t> units = { 64, 4*a_n };
	std::vector<int64_t> units = { 64, a_n };


	ReplayBuffer rb(buffer_size, last_n, es_n, as_n, a_n);

	
	torch::nn::AnyModule qNet(
		md::QNetConv(es_n, as_n, a_n, last_n, 4, 8, 4, dims, pools, units)
	);
	torch::nn::AnyModule qNetTarget(
		md::QNetConv(es_n, as_n, a_n, last_n, 4, 8, 4, dims, pools, units)
	);
	
	/*
	torch::nn::AnyModule qNet(
		md::QNetDRON(es_n, as_n, a_n, last_n, 4, 8, 4, dims, pools, units)
	);
	torch::nn::AnyModule qNetTarget(
		md::QNetDRON(es_n, as_n, a_n, last_n, 4, 8, 4, dims, pools, units)
	);
	*/


	std::vector<int64_t> units_f = { 32, 32 };
	std::vector<int64_t> units_a = { 32, 32 };
	std::vector<int64_t> units_s = { 32, 32 };

	torch::nn::AnyModule iCMNet(
		md::ICMNet(
			es_n, as_n, a_n, last_n,
			16, 8,						// s_emb, a_emb
			units_f, units_a, units_s)
	);

	std::vector<at::Tensor> param1, param2, joinedParameters;
	param1 = qNet.ptr()->parameters();
	param2 = iCMNet.ptr()->parameters();
	joinedParameters.reserve(param1.size() + param2.size());
	joinedParameters.insert(joinedParameters.end(), param1.begin(), param1.end());
	joinedParameters.insert(joinedParameters.end(), param2.begin(), param2.end());


	int nparam(0);
	for (size_t i = 0; i < param1.size(); i++) {
		nparam += param1[i].numel();
	}
	std::cout << "Num Of Params: " << nparam << std::endl;


	//torch::nn::AnyModule qNet(md::QNetState(s_n, a_n, last_n));
	//torch::nn::AnyModule qNetTarget(md::QNetState(s_n, a_n, last_n));

	//torch::optim::Adam opt(qNet.ptr()->parameters(), torch::optim::AdamOptions(0.00001));
	//torch::optim::Adam opt(joinedParameters, torch::optim::AdamOptions(0.0001));
	//torch::optim::Adam opt(joinedParameters, torch::optim::AdamOptions(0.0001));

	torch::optim::Adam opt1(param1, torch::optim::AdamOptions(0.00001));
	torch::optim::Adam opt2(param2, torch::optim::AdamOptions(0.00001));

	LinearLRS lrs1(opt1, 100, 1, 0.001, 30 * buffer_size);
	LinearLRS lrs2(opt2, 100, 1, 0.001, 30 * buffer_size);

	//EpsilonGreedy xpa(1, 0.1, 3*buffer_size);
	EpsilonGreedy xpa(1, 0.1, 5000);
	xpa.update();
	//Boltzmann xpa(10, 0.5, 0.99);

	PESampler pes(buffer_size, 0.6, 0.5);

	//PDQN dqn(rb, batch_size, qNet, qNetTarget, opt, xpa, 0, 0.98, 0, pes);
	//DQN dqn(rb, batch_size, qNet, qNetTarget, opt, lrs, xpa, 0.9, 0.995, 0);
	DQNA dqn(rb, batch_size, qNet, qNetTarget, opt1, lrs1, opt2, lrs2, xpa, 0.9, 0.995, 0, iCMNet, 0.1, 0.3, 1);

	GameSession gs(es_n, as_n, a_n, dqn, max_iter);
	gs.start();
	return 0;
}