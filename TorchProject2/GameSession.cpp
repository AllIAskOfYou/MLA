#include "GameSession.h"

GameSession::GameSession(
	int64_t s_n,
	int64_t a_n,
	RLA& rla,
	size_t max_itr
) :
	s_n(s_n),
	a_n(a_n),
	rla(rla),
	max_itr(max_itr)
{
	s = at::zeros({ s_n });
	oa = at::zeros({ 1 });
	r = at::zeros({ 1 });
	aa = at::zeros({ 1 });
};

void GameSession::start() {
	float req;
	if (ps.connect("\\mla-server")) {
		std::cout << "Connected!" << std::endl;
		while (true) {
			readReq = ps.recieveData(&req, 1);
			if (readReq == 0) {
				continue;
			}
			switch ((int)req) {
			case 0:
				return;
				break;
			case 1:
				update();
				break;
			case 2:
				nextAction();
				break;
			case 3:
				nextOAction();
				break;
			}
		}
	}
}

void GameSession::update() {
	// get new state and oponent action
	readS = ps.recieveData(s.data_ptr<float>(), s_n);
	readA = ps.recieveData(oa.data_ptr<float>(), 1);
	readR = ps.recieveData(r.data_ptr<float>(), 1);
	readTS = ps.recieveData(&ts, 1);

	// if bad data, continue
	if (readS <= 0 || readA <= 0 || readR <= 0) return;

	// save new state, actions and reward
	rla.push(s, aa, oa, r);

	// take one update step on the policy / model
	for (int i = 0; i < 1; i++) {
		rla.update();
	}
}

void GameSession::nextAction() {
	// get next action based on current policy and data
	aa = rla.nextAction();
	aa = aa.to(at::kFloat);

	// send new action to the game
	ps.sendData(aa.data_ptr<float>(), aa.numel());
}

void GameSession::nextOAction() {
}
