#pragma once
#include <vector>
#include <iostream>
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

	//Initialize using already made 2d vector
	Matrix(const vector<vector<float>>& Vals,const int rows,const int cols);

	//Initialize using already made 1d vector
	Matrix(const vector<float>& Vals, const int rows, const int cols);

	//Copy constructor
	Matrix(const Matrix& Mat);

	//returns all zero matrix
	static Matrix Zero(int rows, int cols);

	//returns all one matrix
	static Matrix One(int rows, int cols);

	//Returns values in matrix at specified row and column
	//row 0,column 0 corresponds to upper left value
	float& at(const int row,const int col);

	//Returns a copy of transpose of matrix
	Matrix Transpose()const;


	//Operator overloading
	Matrix operator*(const Matrix& rMat)const;
	Matrix operator*(const float scalar)const;
	Matrix& operator*=(const float scalar);
	Matrix& operator*=(const Matrix& rMat);
	Matrix operator+(const Matrix& rMat)const;
	Matrix& operator+=(const Matrix& rMat);
	Matrix operator-(const Matrix& rMat)const;
	Matrix& operator-=(const Matrix& rMat);
	bool operator==(const Matrix& rMat)const;
	Matrix operator=(const Matrix& rMat);
	friend ostream& operator<<(ostream& os, const Matrix& Mat);
	
	//Return column vector from matrix
	Matrix Column(const int col);
	int get_rows()const;
	int get_cols()const;
	void set_values(vector<vector<float>>& V);
	void set_values(vector<float>& V,bool colVec=true);

};
