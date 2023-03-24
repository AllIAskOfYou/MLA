#pragma once

#include "RLA.h"

class DQN : public RLA {
public:
	DQN(ReplayBuffer& rb);

	void update();
	at::Tensor nextAction();

private:

};

