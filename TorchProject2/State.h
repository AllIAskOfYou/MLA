#pragma once

#include <ATen/ATen.h>

typedef struct State {
	at::Tensor eStates;		// env state
	at::Tensor aStates;		// agent state
	at::Tensor oStates;		// oponent state
	at::Tensor aActions;	// agent action
	at::Tensor oActions;	// oponent action
	at::Tensor joinActions; // joint action

	//at::Tensor nOAtions;	// next opponent action pred

public:
	State inverse() {
		State state;
		state.eStates = eStates;
		state.aStates = oStates;
		state.oStates = aStates;
		state.aActions = oActions;
		state.oActions = aActions;
		state.joinActions = joinActions;
		return state;
	}
} State;