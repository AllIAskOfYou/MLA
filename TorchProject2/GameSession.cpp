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
	if (ps.connect()) {
		std::cout << "Yeay" << std::endl;

		size_t itr = 0;
		for (; itr < max_itr; itr++) {
			// get new state and oponent action
			readS = ps.recieveData(s.data_ptr<float>(), s_n);
			readA = ps.recieveData(oa.data_ptr<float>(), 1);
			readR = ps.recieveData(r.data_ptr<float>(), 1);
			readTS = ps.recieveData(&ts, 1);

			// if bad data, continue
			if (readS <= 0 || readA <= 0 || readR <= 0) continue;

			// save new state, actions and reward
			rla.push(s, aa, oa, r);

			// take one update step on the policy / model
			for (int i = 0; i < 1; i++) {
				rla.update();
			}
			
			// if in terminal state break
			if ((int)ts == 1) break;

			// get next action based on current policy and data
			aa = rla.nextAction();
			aa = aa.to(at::kFloat);
			
			// send new action to the game
			ps.sendData(aa.data_ptr<float>(), aa.numel());
		}
		std::cout << "< Session ended with " << itr << " iterations >" << std::endl;
	}
}
