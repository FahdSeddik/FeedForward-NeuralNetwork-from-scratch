#pragma once
#include "Matrix.h"
#include <string>
/*
				***********************
				*----====Layer====----*
				***********************

This class provides:
	1. Normal Layer 
		At initialization:
			-Specify number of units
			-Specify activation function (Default sigmoid)
			-Optional specify input shape
		Later, it can be initlized/made aware with dimensions from neural network according to previous layers
	2. Dropout Layer: Specify dropout rate
	3. BatchNormalization Layer (To be implemented)
	
*/
class Layer {
private:
	/*
	  [w11 w12]unit 1
	  [w21 w22]unit 2
	*/
	Matrix W;

	/*
		Bias column vector
		[b1]unit 1
		[b2]unit 2
	*/
	Matrix B; 

	Matrix last_input;
	Matrix last_output;
	Matrix delta_W;
	Matrix delta_B;
	//units are rows of matrix
	//input shape defines columns of matrix
	int units,input_shape;
	string activation;
	string wi; //weight initisalization
	//float dropoutRate;

	bool initialized;
	//Generates initial values for a 2d vector
	//using Normalized Xavier Weight Initialization
	void NX(vector<vector<float>>& V);


	//Initializes Bias
	void iB(vector<float>& V);

public:
	//First layer
	//Provide units and input shape with function sigmoid (default)
	//uses Normalized Xavier Weight Initialization (default)
	Layer(const int units, const int input_shape, const string activation = "sigmoid", const string wi = "nx");

	//Normal
	//Provide units with function sigmoid (default)
	Layer(const int units, const string activation = "sigmoid", const string wi = "nx");

	//Copy constructor
	//Shallow copy no need for copy constructor

	//DropOut Layer
	//Layer(const float dropout);

	//BatchNormalization
	//Layer();

	//Takes input_shape if not already given to create matrix
	//uses Normalized Xavier Weight Initialization (default)
	void initialize(const int input_shape);

	//Returns layer result after forward pass
	//Weights*Input + Bias
	Matrix forward(const Matrix& Vector);


	//
	void backward(Matrix& D,float learnRate);
	

	void update_weights();

	int get_units()const;
	int get_input_shape()const;
};