#pragma once
#include "../Dependencies/Layer.h"
#include <string>

class NN {
private:
	//Array of layers
	Layer* Layers;
	int numLayers;
	Matrix last_output;
public:
	//Constructor: Takes array of Layers
	NN(Layer Layers[], int numLayers);

	
	//void fit(float X_train[],const int sizeX,float y_train[],const int sizeY,const int batch_size,const int epochs,const int verbose=0);

	//Performs one forward pass
	//Returns Neural Network output Matrix
	Matrix forward_pass(Matrix& input);

	void back_prop(Matrix& Expected,float learnRate=0.01);
	//void compile(const string optimizer = "adam", const string loss = "mae", const string metrics = "mae");

	//takes a matrix X
	//[Input1column  Input2column....]
	//matrix Y
	//[Output1column  output2column...]
	void train(Matrix& X, Matrix& Y,const int batch_size,const int epochs,const float learnRate);
	void test(Matrix& X);
};
