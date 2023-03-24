#include "GameSession.h"

using namespace torch::indexing;

GameSession::GameSession(
	int64_t s_n,
	int64_t a_n,
	int64_t batch_size,
	int64_t last_n,
	torch::nn::AnyModule oaModel,
	std::shared_ptr<torch::optim::Optimizer> oaModelOptimizer
) :
	s_n(s_n),
	a_n(a_n),
	batch_size(batch_size),
	last_n(last_n),
	oaModel(oaModel),
	oaModelOptimizer(oaModelOptimizer)
{
	states = DTensor({ batch_size + 1, s_n, last_n }, floatOptions);
	aActions = DTensor({ batch_size + 1, last_n }, intOptions);
	oActions = DTensor({ batch_size + 1, last_n }, intOptions);
	rewards = DTensor({ batch_size + 1, last_n }, intOptions);

	s = torch::zeros({ s_n });
	oa = torch::zeros({ 1 });
	r = torch::zeros({ 1 });
	next_aa = torch::zeros({ 1 });
};

void GameSession::start() {
	if (ps.connect()) {
		std::cout << "Yeay" << std::endl;

		while (true) {
			// get new state and oponent action
			readS = ps.recieveData(s.data_ptr<float>(), s_n);
			readA = ps.recieveData(oa.data_ptr<float>(), 1);
			readR = ps.recieveData(r.data_ptr<float>(), 1);

			// if bad data, continue
			if (readS <= 0 || readA <= 0 || readR <= 0) {
				continue;
			}

			// save new state and actions - prepare data for learning
			states.push(s);
			aActions.push(next_aa.to(intOptions));
			oActions.push(oa.to(intOptions));
			rewards.push(r.to(intOptions));

			// update oponent action model
			//updateOaModel();

			next_oa = torch::zeros({ a_n });//getNextOa();
			std::cout << next_oa << std::endl;
			next_oa = next_oa.argmax().to(floatOptions);

			float message = 2;
			//ps.sendData(next_oa.data_ptr<float>(), 1);
			ps.sendData(next_oa.data_ptr<float>(), next_oa.numel());
		}
	}
}

// update oa model on: s_n, aa_n, oa_n -> oa_(n+1)
/*
void GameSession::updateOaModel() {
	oaModelOptimizer->zero_grad();
	torch::Tensor output = oaModel.forward(
		states.index({ Slice(0, -1) }),
		aActions.index({ Slice(0, -1) }),
		oActions.index({ Slice(0, -1) })
	);
	torch::Tensor loss = torch::nn::CrossEntropyLoss()(
		output,
		oActions.index({Slice(1), Slice(), -1})
		);
	loss.backward();
	oaModelOptimizer->step();
}

torch::Tensor GameSession::getNextOa() {
	torch::NoGradGuard no_grad;
	return oaModel.forward(
		states.index({ Slice(-1) }),
		aActions.index({ Slice(-1) }),
		oActions.index({ Slice(-1) })
	);
}
*/