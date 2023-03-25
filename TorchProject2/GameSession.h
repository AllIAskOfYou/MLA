#pragma once

#include <ATen/ATen.h>
#include <iostream>
#include "PipeServer.h"
#include "RLA.h"

// GameSession
// Parameters:
// s_n -> number of features in state
// a_n -> number of actions
// rla -> reinforcement learning algorithm

class GameSession {
public:
	GameSession(
		int64_t s_n,
		int64_t a_n,
		RLA& rla,
		size_t max_itr
	);

	void start();

private:
	// states dimension or number of features
	int64_t s_n;
	// number of actions
	int64_t a_n;
	// reinforcement learning algorithm
	RLA& rla;
	// max number of iterations before terminating
	size_t max_itr;

	// buffers for single data
	at::Tensor s;
	at::Tensor aa;
	at::Tensor oa;
	at::Tensor r;
	float ts;
	at::Tensor next_oa;

	// number of units read
	int readS, readA, readR, readTS;

	// pipe server for comunication with a game process
	PipeServer ps;
};