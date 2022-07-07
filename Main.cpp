#include <iostream>
#include <Windows.h>
using namespace std;
#include "Neural Network/NeuralNetwork.h"
int main() {

	//Initialize an array of layers
	Layer Layers[] = {
		Layer(2,2,"relu","nx"),
		Layer(1,"relu")
	};
	//Initialize neural network
	NN model(Layers, 2);
	cout << "\n\n";

	//Prepare data set
	Matrix X(2, 4), Y(1, 4);
	X.at(0, 0) = 0;
	X.at(1, 0) = 0;
	X.at(0, 1) = 1;
	X.at(1, 1) = 0;
	X.at(0, 2) = 0;
	X.at(1, 2) = 1;
	X.at(0, 3) = 1;
	X.at(1, 3) = 1;

	Y.at(0, 0) = 1;
	Y.at(0, 1) = 1;
	Y.at(0, 2) = 1;
	Y.at(0, 3) = 0;
	int epochs = 10000;
	float learnRate = 0.05;

	cout <<"X_train:\n"<< X <<"Y_train:\n" << Y<<endl;
	//Train 
	model.train(X, Y, epochs, learnRate);
	
	//Test
	cout << "\nTesting:\n";
	Matrix test;
	test = X.Column(0);
	cout << "Y0:";
	model.test(test);
	test = X.Column(1);
	cout << "Y1:";
	model.test(test);
	test = X.Column(2);
	cout << "Y2:";
	model.test(test);
	test = X.Column(3);
	cout << "Y3:";
	model.test(test);
	cout << "\nExpected to be close to: " <<Y;
	
	//Printing summary of model
	model.summary();
}