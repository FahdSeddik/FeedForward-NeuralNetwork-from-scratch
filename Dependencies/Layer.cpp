#include "Layer.h"
#include <cmath>
#include <random>
#include <ctime>
using namespace std;

void Layer::NX(vector<vector<float>>& V)
{
	if (V.size() == 0 || V[0].size() == 0)
		return;
	int rows = V.size(); //number of units-->  m
	int columns = V[0].size();//number of inputs--> n

	// weight = U [-(sqrt(6)/sqrt(n + m)), sqrt(6)/sqrt(n + m)]

	float lower = -sqrt(6) / sqrt(columns + rows);
	float upper = sqrt(6) / sqrt(columns + rows);

	srand(time(NULL));

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			V[i][j] = (float)(rand() % 1000) / 999.0 * (upper - lower) + lower;
		}
	}

}

void Layer::iB(vector<float>& V)
{
	for (int i = 0; i < (int)V.size(); i++)
	{
		V[i] = 0.01;
	}
}

Layer::Layer(const int units, const int input_shape, const string activation,const string wi)
{
	this->activation = activation;
	this->units = units;
	this->wi = wi;
	initialize(input_shape);
}

Layer::Layer(const int units, const string activation,const string wi)
{
	this->units = units;
	this->activation = activation;
	this->wi = wi;
}

void Layer::initialize(const int input_shape)
{
	this->input_shape = input_shape;
	vector<vector<float>> v(units, vector<float>(input_shape));
	if (wi == "nx")
		NX(v);
	else
		NX(v); //to be changed
	
	vector<float> b(units);
	iB(b);
	B.set_values(b);
	W.set_values(v);
	cout << "Initialized Bias:\n" << B;
	cout << "Initialized Weights:\n" << W;
}

Matrix Layer::forward(const Matrix& Vector) const
{
	//Vector is the column vector of inputs to the layer
	//we need to perform the operation Weights*Input + Bias
	Matrix Operation(W*Vector + B);
	if (activation == "sigmoid") {
		for (int i = 0; i < units; i++)
		{
			//	sigmoid(x) = 1/(1+exp(-x))
			Operation.at(i, 0) = (float)1.0 / (float)((float)1.0 + exp(-Operation.at(i, 0)));
		}
	}
	return Operation;
}

int Layer::get_units() const
{
	return units;
}

int Layer::get_input_shape() const
{
	return input_shape;
}
