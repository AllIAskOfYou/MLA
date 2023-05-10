#pragma once

#include <ATen/ATen.h>

typedef struct State {
	at::Tensor eStates;		// env state
	at::Tensor aStates;		// agent state
	at::Tensor oStates;		// oponent state
	at::Tensor aActions;	// agent action
	at::Tensor oActions;	// oponent action
} State;

typedef struct RBSample {
	State states;			// states				(t)
	at::Tensor rewards;		// reward				(t)
	State nStates;			// next states			(t+1)
} RBSample;