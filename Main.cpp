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
		Layer(2,2,"no","nx"),
		Layer(3,"no"),
		Layer(1,"no")
	};
	NN model(Layers, 3);
	cout << "\n\n";
	vector<float> o(1);
	o[0] = 1;
	Matrix Y(o, 1, 1);
	Matrix output = model.forward_pass(Input);
	Matrix first = output;
	for (int i = 0; i < 10; i++)
	{
		cout << "After" << i + 1 << " back prop output:\n";
		model.back_prop(output, Y, 0.2);
		output = model.forward_pass(Input);
		
		cout << output<<"\n";
	}
	cout << "\nFirst:" << first;
	cout << "Last:" << output;
	
}