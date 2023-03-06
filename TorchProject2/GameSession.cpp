#include "GameSession.h"

using namespace torch::indexing;

GameSession::GameSession(
	int64_t s_n,
	int64_t a_n,
	int64_t batch_size,
	int64_t last_n,
	torch::nn::AnyModule oaModel,
	torch::nn::AnyModule envModel,
	std::shared_ptr<torch::optim::Optimizer> oaModelOptimizer,
	std::shared_ptr<torch::optim::Optimizer> envModelOptimizer
) :
	s_n(s_n),
	a_n(a_n),
	batch_size(batch_size),
	last_n(last_n),
	oaModel(oaModel),
	envModel(envModel),
	oaModelOptimizer(oaModelOptimizer),
	envModelOptimizer(envModelOptimizer)
{
	states = DTensor({ batch_size + 1, s_n, last_n }, floatOptions);
	aActions = DTensor({ batch_size + 1, 1, last_n }, intOptions);
	oActions = DTensor({ batch_size + 1, 1, last_n }, intOptions);

	s = torch::zeros({ s_n });
	oa = torch::zeros({ 1 });
	next_aa = torch::zeros({ 1 });
};

void GameSession::start() {
	if (ps.connect()) {
		std::cout << "Yeay" << std::endl;

		while (true) {
			// get new state and oponent action
			readS = ps.recieveData(s.data_ptr<float>(), 3);
			readA = ps.recieveData(oa.data_ptr<float>(), 1);

			// if bad data, continue
			if (readS <= 0 || readA <= 0) {
				continue;
			}

			// save new state and actions - prepare data for learning
			states.push(s);
			//aActions.push(torch::one_hot(next_aa.to(intOptions), a_n).squeeze());
			//oActions.push(torch::one_hot(oa.to(intOptions), a_n).squeeze());
			aActions.push(next_aa.to(intOptions));
			oActions.push(oa.to(intOptions));

			// update oponent action model and environment model
			updateOaModel();
			updateEnvModel();


			//std::cout << states.toTensor() << std::endl;
			//std::cout << aActions.toTensor() << std::endl;
			//std::cout << oActions.toTensor() << std::endl;

			//std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
			//std::cout << states.toTensor() << std::endl;

			next_oa = getNextOa();
			std::cout << next_oa << " <-----> " << next_oa.detach() << std::endl;
			next_oa = next_oa
				.argmax();
			next_aa = next_oa.clone();
			next_oa = next_oa.to(torch::TensorOptions().dtype(torch::kFloat32));

			std::cout << next_aa << " :::: " << next_oa << std::endl;

			/*
			envModelOptimizer->zero_grad();
			torch::Tensor next_s = envModel.forward(
				states.a.index({ Slice(-2, -1) }),
				aActions.a.index({ Slice(-1) }),
				oActions.a.index({ Slice(-1) })
			);
			std::cout << next_s << " : " << states.a.index({ Slice(-1) }) << std::endl;
			*/
			float message = 2;
			//ps.sendData(next_oa.data_ptr<float>(), 1);
			ps.sendData(next_oa.data_ptr<float>(), next_oa.numel());
		}
	}
}

// update oa model on: s_n, aa_n, oa_n -> oa_(n+1)
void GameSession::updateOaModel() {
	oaModelOptimizer->zero_grad();
	torch::Tensor output = oaModel.forward(
		states.a.index({ Slice(0, -1) }),
		aActions.a.index({ Slice(0, -1) }),
		oActions.a.index({ Slice(0, -1) })
	);
	torch::Tensor loss = torch::nn::CrossEntropyLoss()(
		output,
		oActions.a.index({Slice(1), Slice(), -1})
		);
	loss.backward();
	oaModelOptimizer->step();
}

// update env model on: s_n, aa_(n+1), oa_(n+1) -> s_(n+1)
void GameSession::updateEnvModel() {
	envModelOptimizer->zero_grad();
	torch::Tensor output = envModel.forward(
		states.a.index({ Slice(0, -1) }),
		aActions.a.index({ Slice(1) }),
		oActions.a.index({ Slice(1) })
	);
	torch::Tensor loss = torch::nn::L1Loss()(
		output,
		states.a.index({ Slice(1), Slice(), -1 })
		);
	loss.backward();
	envModelOptimizer->step();
}

torch::Tensor GameSession::getNextOa() {
	torch::NoGradGuard no_grad;
	return oaModel.forward(
		states.a.index({ Slice(-1) }),
		aActions.a.index({ Slice(-1) }),
		oActions.a.index({ Slice(-1) })
	);
}

torch::Tensor GameSession::decideNextAction() {
	torch::NoGradGuard no_grad;
	torch::Tensor oa_prob = oaModel.forward(
		states.a.index({ Slice(-1) }),
		aActions.a.index({ Slice(-1) }),
		oActions.a.index({ Slice(-1) })
	).softmax(1, torch::kFloat32)[0];
	torch::Tensor next_states;

	torch::Tensor s = states.a[0].repeat({ a_n });

	for (int i = 0; i < a_n; i++) {
		next_states = envModel.forward(
			states.a.index({ Slice(-1) }),
			torch::full({ 1, a_n }, i),

		);
	}

	return torch::zeros({1,2});
}