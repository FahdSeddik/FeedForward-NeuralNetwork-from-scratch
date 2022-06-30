#include <iostream>
using namespace std;
#include "Neural Network/NeuralNetwork.h"

int main() {
	vector<float> i(2);
	i[0] = 1;
	i[1] = 0;
	Matrix Input(i,2,1);
	Layer Layers[] = {
		Layer(2,2,"sigmoid","nx"),
		Layer(1)
	};
	NN model(Layers, 2);
	cout << "\nInput:\n" << Input;
	cout <<"Final result:\n" << model.forward_pass(Input);
}