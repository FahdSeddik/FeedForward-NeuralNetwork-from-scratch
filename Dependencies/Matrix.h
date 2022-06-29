#pragma once
#include <vector>
using namespace std;
/*
				************************
				*----====Matrix====----*
				************************

Class matrix provides simple matrix operations and declarations

*/
class Matrix {
private:
	int _rows, _cols;
	/*
	Example: Matrix 6x7

	 0  1  2  3	 4  5  6
	 7  8  9 10 11 12 13
	14 15 16 17 18 19 20
	21 22 23 24 25 26 27
	28 29 30 31 32 34 35
	36 37 38 39 40 41 42
	
	

	*/
	vector<vector<float>> values;
public:
	//Takes number of rows and columns for matrix
	Matrix(const int rows=0,const int cols=0);

	//Initialize using already made vector
	Matrix(const vector<vector<float>>& Vals,const int rows,const int cols);


	//Copy constructor
	Matrix(const Matrix& Mat);



	//Returns values in matrix at specified row and column
	//row 0,column 0 corresponds to upper left value
	float at(const int row,const int col)const;


	//Operator overloading
	Matrix operator*(const Matrix& rMat)const;
	Matrix& operator*=(const Matrix& rMat);
	Matrix operator+(const Matrix& rMat)const;
	Matrix& operator+=(const Matrix& rMat);
	Matrix operator-(const Matrix& rMat)const;
	Matrix& operator-=(const Matrix& rMat);
	bool operator==(const Matrix& rMat)const;
	
};
