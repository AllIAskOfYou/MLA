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
		int64_t es_n,
		int64_t as_n,
		int64_t a_n,
		RLA& rla,
		size_t max_itr
	);

	void start();

private:
	void push();
	void update();
	void nextAction();
	void selfPlay();
	void reset();

private:
	// states dimension or number of features
	int64_t es_n;
	int64_t as_n;
	// number of actions
	int64_t a_n;
	// reinforcement learning algorithm
	RLA& rla;
	// max number of iterations before terminating
	size_t max_itr;

	// buffers for single data
	at::Tensor es;
	at::Tensor as;
	at::Tensor os;
	at::Tensor aa_out;
	at::Tensor aa;
	at::Tensor oa;
	at::Tensor r;
	at::Tensor t;

	// number of units read
	int readReq, readES, readAS, readOS, readAA, readOA, readR, readT;

	// pipe server routes for comunication with a game process
	PipeServer ps;
};