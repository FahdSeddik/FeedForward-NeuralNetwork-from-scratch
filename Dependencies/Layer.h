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
	Matrix m;
	int units,input_shape;
	string activation;
	float dropoutRate;
public:
	//First layer
	//Provide units and input shape with function sigmoid (default)
	Layer(const int units, const int input_shape, const string activation = "sigmoid");

	//Normal
	//Provide units with function sigmoid (default)
	Layer(const int units, const string activation = "sigmoid");


	//DropOutLayer
	Layer(const float dropout);





};