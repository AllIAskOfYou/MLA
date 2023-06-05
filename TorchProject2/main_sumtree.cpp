#include "main_sumtree.h"
#include "SumTree.h"
#include "MaxTree.h"
#include <iostream>
#include "ATen/ATen.h"

int main_sumtree() {
	std::srand(0);

	SumTree st(4);
	st.push(1);
	st.push(1);
	st.update(-1, 5);
	st.push(1);
	
	//std::vector<int64_t> indexes = {0, 1, 2, 3};
	auto indexes = st.sample_batch(4);
	at::Tensor idx = at::tensor({ 0, 1, 2, 3 }, at::TensorOptions().dtype(c10::ScalarType::Long));
	std::cout << "asd" << std::endl;
	auto pes = st.get(indexes);
	st.update_batch(indexes, pes+1.2);


	std::cout << pes << std::endl;
	std::cout << st.get(idx) << std::endl;
	/*
	for (int i = 0; i < pes.size(); i++) {
		std::cout << pes[i] << ", ";
	}
	std::cout << std::endl;

	auto smpl = st.sample_batch(2);
	auto vals = st.get(smpl);

	at::Tensor tensor = at::from_blob(vals.data(), (int)vals.size());
	vals[0] = 3.4;
	

	std::cout << tensor << std::endl;
	
	for (int i = 0; i < vals.size(); i++) {
		std::cout << vals[i] << ", ";
	}
	std::cout << std::endl;

	std::cout << st.sample(7) << std::endl;

	std::cout << st.get_value() << std::endl;

	MaxTree mt(4);
	mt.push(1);
	mt.push(1);
	mt.update(-1, 5);
	mt.push(1);

	std::cout << "Max: " << mt.get_value() << std::endl;
	*/
	return 0;
}