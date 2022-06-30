#pragma once
#include "../Dependencies/Layer.h"
#include <string>

class NN {
private:
	//Array of layers
	Layer* Layers;
	int numLayers;
public:
	//Constructor: Takes array of Layers
	NN(Layer Layers[], int numLayers);

	
	//void fit(float X_train[],const int sizeX,float y_train[],const int sizeY,const int batch_size,const int epochs,const int verbose=0);

	//Performs one forward pass
	//Returns Neural Network output Matrix
	Matrix forward_pass(Matrix& input);


	//void compile(const string optimizer = "adam", const string loss = "mae", const string metrics = "mae");
};
