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
		Layer(2,2,"sigmoid","nx"),
		Layer(3),
		Layer(1,"no")
	};
	NN model(Layers, 3);
	cout << "\nInput:\n" << Input;
	Matrix output = model.forward_pass(Input);
	cout << "Final result:\n" << output;
	cout << "\n\n";
	vector<float> o(1);
	o[0] = 1;
	Matrix Y(o, 1, 1);
	Matrix first = output;
	for (int i = 0; i < 25; i++)
	{
		model.back_prop(output, Y, 0.1);
		//cout << "After" << i + 1 << " back prop output:\n";
		output = model.forward_pass(Input);
		
		//cout << output<<"\n";
	}
	cout << "\nFirst:" << first;
	cout << "Last:" << output;
	
}