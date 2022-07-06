#include <iostream>
#include <Windows.h>
using namespace std;
#include "Neural Network/NeuralNetwork.h"

int main() {
	vector<float> i(2);
	i[0] = 0;
	i[1] = 1;
	Matrix Input(i,2,1);
	Layer Layers[] = {
		Layer(2,2,"nx"),
		Layer(1),
		Layer(1)
	};
	NN model(Layers, 3);
	cout << "\n\n";
	Matrix X(2,4),Y(1,4);
	X.at(0, 0) = 0;
	X.at(1, 0) = 0;
	X.at(0, 1) = 1;
	X.at(1, 1) = 0;
	X.at(0, 2) = 0;
	X.at(1, 2) = 1;
	X.at(0, 3) = 1;
	X.at(1, 3) = 1;
	Y.at(0, 0) = 0;
	Y.at(0, 1) = 1;
	Y.at(0, 2) = 1;
	Y.at(0, 3) = 0;
	Matrix X_test(2,1);
	cout << "First:\n";
	X_test.at(0, 0) = 0;
	X_test.at(1, 0) = 0;
	model.test(X_test);
	X_test.at(0, 0) = 1;
	X_test.at(1, 0) = 0;
	model.test(X_test);
	X_test.at(0, 0) = 0;
	X_test.at(1, 0) = 1;
	model.test(X_test);
	X_test.at(0, 0) = 1;
	X_test.at(1, 0) = 1;
	model.test(X_test);
	model.train(X, Y, 1, 500, 0.1);
	cout << "\n\nAfter:\n";
	X_test.at(0, 0) = 0;
	X_test.at(1, 0) = 0;
	model.test(X_test);
	X_test.at(0, 0) = 1;
	X_test.at(1, 0) = 0;
	model.test(X_test);
	X_test.at(0, 0) = 0;
	X_test.at(1, 0) = 1;
	model.test(X_test);
	X_test.at(0, 0) = 1;
	X_test.at(1, 0) = 1;
	model.test(X_test);
}