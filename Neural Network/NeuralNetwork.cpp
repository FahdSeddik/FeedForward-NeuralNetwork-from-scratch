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

void NN::back_prop(Matrix& MSE,float learnRate)
{
	//Y is a column vector
	//rows are units of output layer
	
	// MSE cost function (y'-y)^2
	// delta = 2*(y'-y)*del wj
	Matrix Delta(MSE);
	for (int i = numLayers - 1; i >= 0; i--)
	{
		Layers[i].backward(Delta, learnRate);
	}
	for (int i = 0; i < numLayers; i++)
	{
		Layers[i].update_weights();
	}
}

void NN::train(Matrix& X, Matrix& Y,const int batch_size,const int epochs,const float learnRate)
{
	Matrix Input,Output;
	MSE = Matrix::Zero(Y.get_rows(), 1);
	int col = 0,upperbound=batch_size;
	while (col<=Y.get_cols())
	{
		for (int i = 0; i < epochs; i++)
		{
			for (int j = col; j <= col+batch_size; j++)
			{
				if (j >= Y.get_cols())
					break;
				Input = X.Column(j);
				Output = forward_pass(Input);
				for (int k = 0; k < MSE.get_rows(); k++)
				{
					MSE.at(k, 0) += 2 * (Y.at(k, j) - Output.at(k, 0));
				}
			}
			MSE = MSE * (float)((float)1.0 / (float)batch_size);
			back_prop(MSE, learnRate);
			MSE -= MSE;
		}
		col += batch_size;
		//upperbound = (col + batch_size) >= Y.get_cols() ? (Y.get_cols()) : (col + batch_size);
	}
}

void NN::test(Matrix& X)
{
	cout << forward_pass(X);
}
