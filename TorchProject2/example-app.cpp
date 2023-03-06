// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <torch/torch.h>
#include <iostream>
#include <chrono>

struct SoftmaxRegressionImpl : torch::nn::Module {
	SoftmaxRegressionImpl(int in_dim, int out_dim)
		: lin(torch::nn::LinearOptions(in_dim, out_dim))
	{
		register_module("lin", lin);
	}

	torch::Tensor forward(torch::Tensor x) {
		x = lin(x);
		return x;
	}

	torch::nn::Linear lin;
};
TORCH_MODULE(SoftmaxRegression);

/*
int main()
{
	torch::Tensor x = torch::zeros({ 16, 1 });
	torch::Tensor y = torch::zeros({ 16 }, torch::TensorOptions().dtype(torch::kInt64));

	y.index({ torch::indexing::Slice(1, torch::indexing::None, 2) }) = 1;
	SoftmaxRegression model(1, 3);

	torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(1));
	//torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1));
	torch::nn::CrossEntropyLoss criterion;
	// mogoce je to zgoraj problem ce je kaj narobe !!!!!!
	uint8_t m = 5;

	auto start = std::chrono::steady_clock::now();
	auto end = std::chrono::steady_clock::now();
	for (uint8_t i = 0; i < m; i++) {
		start = std::chrono::steady_clock::now();
		//model->zero_grad();
		optimizer.zero_grad();
		torch::Tensor output = model->forward(x);
		torch::Tensor loss = criterion(output, y);
		loss.backward();
		optimizer.step();
		end = std::chrono::steady_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
	}
	torch::Tensor pred = model->forward(x);
	std::cout << pred.softmax(1) << std::endl;
	//auto end = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	start = std::chrono::steady_clock::now();
	//for (uint8_t i = 0; i < m; i++) {
	model->zero_grad();
	torch::Tensor output = model->forward(x);
	torch::Tensor loss = criterion(output, y);
	loss.backward();
	optimizer.step();
	//}
	end = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	start = std::chrono::steady_clock::now();
	torch::NoGradGuard no_grad;
	pred = model->forward(x);
	end = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;

	std::cout << pred.softmax(1) << std::endl;

	std::cout << "Hello World!\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
*/
