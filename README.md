# Feed-Forward Neural Network from scratch
This is a FeedForward Neural Network with back-propagation written in C++ with no external libraries. The aim for this project was to create a simple version of TensorFlow's Keras sequential neural network from scratch in C++.

## Classes

In Dependencies folder:
* **Matrix**: This is a simple matrix class that allows for matrix operations and several other useful methods.
  - at(row,col) : returns value at specified row and column (0-indexed).
  - Transpose() : returns a transposed copy of current matrix.
  - static Matrix::Zero(rows,cols) **and** Matrix::One(rows,cols) : return a matrix of all zeroes or ones respectively provided number of rows and columns.
  - set_values(vector,rows,cols) : initialize a matrix with a 2d vector of floats (can accept 1d vector of floats).  <br></br>
* **Layer**: Basic element of the NN class. It provides a handful of activation functions.
  * Activation Functions:
    - Sigmoid **(Default)** 
    - ReLU
  * Weight Initialization:
    - Normalized Xavier
  * Some Methods:
    - forward(Matrix) : returns the output of activation(Weights*Matrix + Bias) as a Matrix.
    - backward(Matrix,float) : calculates delta_W and delta_B for both weights and biases respectively.
    - summary() : prints a summary of the layer properties (units, input shape, etc.).

### Neural Network:
* Methods:
  - forward_pass(Matrix) : returns output of one forward pass using the neural network given Matrix input.
  - back_prop(Matrix,float) : performs one backward pass and updates weights and biases of all layers given expected Matrix and learnRate float.
  - train(Matrix,Matrix,int,float) : trains neural network given X_train and y_train matrices provided number of epochs and learnRate. (Please refer to data set preparation section for more details)
  - test(Matrix) : prints neural network output after 1 pass given input. (equivalent to printing forwad_pass).
  - summary() : prints a summary of all the neural network properties. (shows weight and biases matrices of each layer).
  
## Data Set Preparation

In order to train our model we would require a data set. The method train() expects 2 **Matrix** instances.
Each of the matrices should be as follows.
```
[Column_1  Column_2   ...]
```
For the matrix X Each column would represent one input to the neural network (a bit counter intuitive, I know :smile:). On the other hand, it would represent the expected output for y.  <br></br>
**Note: when it comes to test() method it should be provided with a Column vector instance of class Matrix.**

## Instructions

To initialize a neural network:
```cpp
#include "NeuralNetwork/NeuralNetwork.h"

int main(){
  //Create array of layers
  // Layer(units, input_shape, activation, weight_initialization)
  // all strings are typed in **lower-case**
  // You need to only specify input shape in first layer
  Layer Layers[] = {
		Layer(2,2,"relu","nx"),
		Layer(1,"relu")
  }
  //Initialize neural network with array and size
  NN model(Layers, 2);
}
```
This would create an architecture shown in the picture below.  <br></br>
![image](https://user-images.githubusercontent.com/62207434/177870738-5a0eb6d2-db86-46fb-8140-ff5c596a44d1.png)
<br></br>
After that, you would want to prepare a dataset. A way you could do that is by doing the following:
```cpp
  Matrix X(2, 4), Y(1, 4);
  X.at(0, 0) = 0;
  X.at(1, 0) = 0;
  X.at(0, 1) = 1;
  X.at(1, 1) = 0;
  X.at(0, 2) = 0;
  X.at(1, 2) = 1;
  X.at(0, 3) = 1;
  X.at(1, 3) = 1; 
  
  Y.at(0, 0) = 1;
  Y.at(0, 1) = 1;
  Y.at(0, 2) = 1;
  Y.at(0, 3) = 0;
```
This would create the equivalent matrices shown below.  <br></br>
![image](https://user-images.githubusercontent.com/62207434/177877035-60a6b945-a767-450d-bfbe-eefa8cf4ae1a.png)

If you noticed, it acts as the logical table of a NAND gate!.  
Now, we want to train our model. We cant use the train() method.
```cpp
  int epochs = 10000;
  float learnRate = 0.05;
  //Display train data set
  cout <<"X_train:\n"<< X <<"Y_train:\n" << Y<<endl;
  //Train 
  model.train(X, Y, epochs, learnRate);
```
You would see this on your screen. It will also print a progress bar that would gradually print 10 "#" hashes, each representing 10% closer to completion.  <br></br>
<img src="https://user-images.githubusercontent.com/62207434/177878400-f2ef97d2-2331-4f5b-89f2-0c5c3754cbbc.gif" width="250" /> <br></br>
When it comes to testing, all you need is have column vectors of your input then use test() method.
```cpp
  //Test
  cout << "\nTesting:\n";
  Matrix test;
  test = X.Column(0);
  cout << "Y0:";
  model.test(test);
  test = X.Column(1);
  cout << "Y1:";
  model.test(test);
  test = X.Column(2);
  cout << "Y2:";
  model.test(test);
  test = X.Column(3);
  cout << "Y3:";
  model.test(test);
  cout << "\nExpected to be close to: " <<Y;
```
Then, you should have something like this.  <br></br>
![image](https://user-images.githubusercontent.com/62207434/177879211-4f69494b-b851-4cff-8ee0-bccb52328244.png) <br></br>

If you want, you could print a model's summary using the summary() method as shown below.
```cpp
  //Printing summary of model
  model.summary();
```
![image](https://user-images.githubusercontent.com/62207434/177879631-9b71049f-3be6-4655-b42e-b47708426f73.png)

## Taks
- [ ] Implement Dropout layers.
- [ ] Implement BatchNormalization layers.
- [ ] Add ability to change cost function.
- [ ] Add Adam optimizer.
- [ ] Optimize matrix multiplication. (maybe use Strassen's Algorithm)
