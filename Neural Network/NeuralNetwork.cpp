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
		//cout <<"Operation num: "<<i+1<<" out of "<< numLayers<<"\n" << Output;
	}

	return Output;
}

void NN::back_prop(Matrix& Output,Matrix& Y,float learnRate)
{
	//Y is a column vector
	//rows are units of output layer
	
	// MSE cost function (y'-y)^2
	// delta = 2*(y'-y)*del wj
	Matrix delta(Y);

	for (int i = 0; i < Y.get_rows(); i++)
	{
		delta.at(i, 0) = 2 * (Y.at(i, 0) - Output.at(i, 0));
	}

	for (int i = numLayers - 1; i >= 0; i--)
	{
		Layers[i].backward(delta, learnRate);
	}
	for (int i = 0; i < numLayers; i++)
	{
		Layers[i].update_weights();
	}
}

void NN::train(Matrix& X, Matrix& Y)
{

}
