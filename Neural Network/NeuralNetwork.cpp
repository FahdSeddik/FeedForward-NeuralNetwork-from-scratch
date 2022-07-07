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
	
	
	last_output = Output;
	return Output;
}

void NN::back_prop(Matrix& Expected,float learnRate)
{
	//Y is a column vector
	//rows are units of output layer
	
	// MSE cost function 1/2(y'-y)^2
	// d(y) = 2*1/2*(y'-y)*
	
	Matrix Delta = (Expected - last_output)*2;
	

	for (int i = numLayers - 1; i >= 0; i--)
	{
		Layers[i].backward(Delta, learnRate);
	}
	for (int i = 0; i < numLayers; i++)
	{
		Layers[i].update_weights();
	}
}

void NN::train(Matrix& X, Matrix& Y,const int epochs,const float learnRate)
{
	cout << "Training model...\n";
	Matrix Input,Expected;
	int col = 0;
	cout << "Progress:";
	for (int i = 0; i < epochs; i++)
	{
		//pass whole train set per epoch
		for (int j = 0; j < Y.get_cols(); j++)
		{
			Input = X.Column(j);
			forward_pass(Input);
			Expected = Y.Column(j);
			back_prop(Expected,learnRate);
		}
		if (i % (epochs / 10) == 0)
			cout << "#";
	}
	cout << "\nTrained Successfully.\n";
}

void NN::test(Matrix& X)
{
	cout << forward_pass(X);
}

void NN::summary() const
{
	cout << "\n\t--===Model Summary===--\n";
	cout << "\t**Number of Layers: " << numLayers<<"**\n";
	for (int i = 0; i < numLayers; i++)
	{
		cout << "-=Layer number: " << i + 1 << endl;
		Layers[i].summary();
		cout << endl;
	}
}
