#include "GameSession.h"
#include <chrono>

GameSession::GameSession(
	int64_t es_n,
	int64_t as_n,
	int64_t a_n,
	RLA& rla,
	size_t max_itr
) :
	es_n(es_n),
	as_n(as_n),
	a_n(a_n),
	rla(rla),
	max_itr(max_itr)
{
	es = at::zeros({ es_n });
	as = at::zeros({ as_n });
	os = at::zeros({ as_n });
	aa = at::zeros({ 1 });
	oa = at::zeros({ 1 });
	r = at::zeros({ 1 });
};

void GameSession::start() {
	float req;
	int d = 0;
	float count = 0;
	if (!ps.connect("\\mla-server")) return;

	std::cout << "Connected!" << std::endl;
	std::chrono::steady_clock::time_point t0, t1;
	std::chrono::microseconds d1;
	while (true) {
		readReq = ps.recieveData(&req, 1);
		if (readReq == 0) continue;

		switch ((int)req) {
		case 0:
			return;
			break;
		case 1:
			push();
			break;
		case 2:
			t0 = std::chrono::high_resolution_clock::now();
			update();
			t1 = std::chrono::high_resolution_clock::now();
			d1 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
			d += d1.count() / 1000;
			count += 1;
			//std::cout << "Update time: \n" << d / count << std::endl
			break;
		case 3:
			nextAction();
			break;
		case 4:
			selfPlay();
			break;
		}
	}
}

void GameSession::push() {
	// get new state and oponent action
	readES = ps.recieveData(es.data_ptr<float>(), es_n);
	readAS = ps.recieveData(as.data_ptr<float>(), as_n);
	readOS = ps.recieveData(os.data_ptr<float>(), as_n);
	readA = ps.recieveData(oa.data_ptr<float>(), 1);
	readR = ps.recieveData(r.data_ptr<float>(), 1);

	// if bad data, continue
	if (readES <= 0 || readAS <= 0 || readOS <= 0 || readA <= 0 || readR <= 0) return;

	// save new state, actions and reward
	rla.push(es, as, os, aa, oa, r);
}

void GameSession::update() {
	// take one update step on the policy / model
	auto t0 = std::chrono::high_resolution_clock::now();
	rla.update();
	auto t1 = std::chrono::high_resolution_clock::now();
	auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
	//std::cout << "update time: \n" << d1.count() << std::endl;
}

void GameSession::nextAction() {
	// get next action based on current policy and data
	aa = rla.nextAction();
	aa = aa.to(at::kFloat);

	// send new action to the game
	ps.sendData(aa.data_ptr<float>(), aa.numel());
}

void GameSession::selfPlay() {
	// get next self opponent action
	oa = rla.selfPlay();
	oa = oa.to(at::kFloat);

	// send new action to the game
	ps.sendData(oa.data_ptr<float>(), oa.numel());
}
