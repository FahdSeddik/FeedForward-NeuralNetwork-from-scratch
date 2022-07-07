#include <iostream>
#include <Windows.h>
#include <fstream>
using namespace std;
#include "Neural Network/NeuralNetwork.h"
int main() {
	Layer Layers[] = {
		Layer(2,2,"relu","nx"),
		Layer(1,"relu")
	};
	NN model(Layers, 2);
	cout << "\n\n";
	Matrix X0(2, 1), X1(2, 1), X2(2, 1), X3(2, 1),Y0(1,1), Y1(1, 1), Y2(1, 1), Y3(1, 1);
	X0.at(0, 0) = 0;
	X0.at(1, 0) = 0;
	X1.at(0, 0) = 1;
	X1.at(1, 0) = 0;
	X2.at(0, 0) = 0;
	X2.at(1, 0) = 1;
	X3.at(0, 0) = 1;
	X3.at(1, 0) = 1;
	Y0.at(0, 0) = 1;
	Y1.at(0, 0) = 1;
	Y2.at(0, 0) = 1;
	Y3.at(0, 0) = 0;
	Matrix Xs[] = { X0,X1,X2,X3 };
	Matrix Ys[] = { Y0,Y1,Y2,Y3 };
	int epochs = 1000;
	Matrix Output;
	for (int j = 0; j < epochs; j++)
	{
		for (int i = 0; i < 4; i++)
		{
			model.forward_pass(Xs[i]);
			model.back_prop(Ys[i],0.05);
		}
	}
	cout << "\n\n";
	model.test(X0);
	model.test(X1);
	model.test(X2);
	model.test(X3);
}