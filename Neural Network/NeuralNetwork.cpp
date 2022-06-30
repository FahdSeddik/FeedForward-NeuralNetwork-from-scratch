#include "NeuralNetwork.h"

NN::NN(Layer Layers[], int numLayers)
{
	this->numLayers = numLayers;
	this->Layers = Layers;
	for (int i = 1; i < numLayers; i++)
	{
		this->Layers[i].initialize(this->Layers[i - 1].get_units());
	}
}

Matrix NN::forward_pass(Matrix& Input)
{
	Matrix Output(Input);

	for (int i = 0; i < numLayers; i++)
	{
		Output=Layers[i].forward(Output);
		cout <<"Operation num: "<<i+1<<" out of "<< numLayers<<"\n" << Output;
	}

	return Output;
}
