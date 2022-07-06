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
	vector<vector<float>> v(units, vector<float>(input_shape,0.0f));
	delta_W.set_values(v);
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

Matrix Layer::forward(const Matrix& Vector)
{
	last_input = Vector;
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
	last_output = Operation;
	return Operation;
}

void Layer::backward(Matrix& D,float learnRate)
{
	// Matrix D is Delta
		
	if (activation == "sigmoid") {
		delta_W = D * last_output.Transpose() * (Matrix::One(last_output.get_rows(), last_input.get_cols()) - last_output)* last_input.Transpose() * learnRate;

		D = W.Transpose() * D* last_output.Transpose() * (Matrix::One(last_output.get_rows(), last_input.get_cols()) - last_output);
	}
	else {
		delta_W = D * last_input.Transpose() * learnRate;
		D = W.Transpose() * D;
	}
	
}



void Layer::update_weights()
{
	W += delta_W;
	//cout << "Updated by:\n" << delta_W;
	delta_W -=delta_W;
}

int Layer::get_units() const
{
	return units;
}

int Layer::get_input_shape() const
{
	return input_shape;
}
