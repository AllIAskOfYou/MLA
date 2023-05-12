#pragma once

#include <ATen/ATen.h>

typedef struct State {
	at::Tensor eStates;		// env state
	at::Tensor aStates;		// agent state
	at::Tensor oStates;		// oponent state
	at::Tensor aActions;	// agent action
	at::Tensor oActions;	// oponent action

public:
	State inverse() {
		State state;
		state.eStates = eStates;
		state.aStates = oStates;
		state.oStates = aStates;
		state.aActions = oActions;
		state.oActions = aActions;
		return state;
	}
} State;