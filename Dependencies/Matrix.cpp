#include "Matrix.h"

Matrix::Matrix(const int rows,const int cols)
{
	_rows = rows;
	_cols = cols;
	if(rows != 0 && cols !=0)
		values.resize(_rows,vector<float>(_cols,0.0f));
}

Matrix::Matrix(const vector<vector<float>>& Vals,const int rows,const int cols)
{
	if ((int)Vals.size() != rows && (int)Vals[0].size() != cols)
		throw "Matrix Initialization: Incompatible vector given.";
	_cols = cols;
	_rows = rows;
	values = Vals;
}

Matrix::Matrix(const Matrix& Mat)
{
	values = Mat.values;
	_rows = Mat._rows;
	_cols = Mat._cols;
}

float Matrix::at(const int row,const int col) const
{
	/*
	Example: Matrix 6x7

	 0  1  2  3	 4  5  6
	 7  8  9 10 11 12 13
	14 15 16 17 18 19 20
	21 22 23 24 25 26 27
	28 29 30 31 32 34 35
	36 37 38 39 40 41 42
	*/
	if (row<0 || col<0 || row>=_rows || col>=_cols)
		throw "Matrix Accessing: Index out of bounds.";
	return values[row][col];
}

Matrix Matrix::operator*(const Matrix& rMat) const
{
	if (_cols != rMat._rows)
		throw "Matrix Multiplication: Incompatible size.";
	int len = _rows * rMat._cols;
	vector<vector<float>> v(_rows,vector<float>(rMat._cols,0.0f));
	/*
		3x3		3x3

		0 1 2   0 1 2	2x3		0x0+1x3+2x6		0x1+1x4+2x7		0x2+1x5+2x8
		3 4 5   3 4 5   --->	3x0+4x3+5x6		3x1+4x4+5x7		3x2+4x5+5x8
		6 7 8   6 7 8			7x0+6x3+8x6		7x1+6x4+8x7		7x2+6x5+8x8
	*/
	float sum = 0.0;
	for (int i = 0; i < _rows; i++)
	{
		for (int j = 0; j < rMat._cols; j++)
		{
			//row i*col j
			sum = 0.0;
			for (int k = 0; k < _cols; k++)
			{
				sum+=values[i][k] * rMat.values[k][j];
			}
			v[i][j] = sum;
		}
	}
	return Matrix(v,_rows,rMat._cols);
}

Matrix& Matrix::operator*=(const Matrix& rMat)
{
	if (_cols != rMat._rows)
		throw "Matrix Multiplication: Incompatible size.";
	float sum = 0.0;
	vector<vector<float>> v(_rows, vector<float>(rMat._cols, 0.0f));
	for (int i = 0; i < _rows; i++)
	{
		for (int j = 0; j < rMat._cols; j++)
		{
			//row i*col j
			sum = 0.0;
			for (int k = 0; k < _cols; k++)
			{
				sum += values[i][k] * rMat.values[k][j];
			}
			v[i][j] = sum;
		}
	}
	values = v;
	//same rows number
	_cols = rMat._cols;
	return *this;
}

Matrix Matrix::operator+(const Matrix& rMat) const
{
	if (rMat._cols != _cols || rMat._rows != _rows)
		throw "Matrix Addition: Incompatible size.";
	vector<vector<float>> v(_rows, vector<float>(_cols, 0.0f));
	for (int i = 0; i < _rows; i++)
		for (int j = 0; j < _cols; j++)
			v[i][j] = values[i][j] + rMat.values[i][j];

	return Matrix(v,_rows,_cols);
}

Matrix& Matrix::operator+=(const Matrix& rMat)
{
	if (rMat._cols != _cols || rMat._rows != _rows)
		throw "Matrix Addition: Incompatible size.";
	for (int i = 0; i < _rows; i++)
	{
		for (int j = 0; j < _cols; j++)
		{
			values[i][j] += rMat.values[i][j];
		}
	}
	return *this;
}

Matrix Matrix::operator-(const Matrix& rMat) const
{
	if (rMat._cols != _cols || rMat._rows != _rows)
		throw "Matrix Subtraction: Incompatible size.";
	vector<vector<float>> v(_rows, vector<float>(_cols, 0.0f));
	for (int i = 0; i < _rows; i++)
		for (int j = 0; j < _cols; j++)
			v[i][j] = values[i][j] - rMat.values[i][j];
	return Matrix(v, _rows, _cols);
}

Matrix& Matrix::operator-=(const Matrix& rMat)
{
	if (rMat._cols != _cols || rMat._rows != _rows)
		throw "Matrix Subtraction: Incompatible size.";
	for (int i = 0; i < _rows; i++)
	{
		for (int j = 0; j < _cols; j++)
		{
			values[i][j] -= rMat.values[i][j];
		}
	}
	return *this;
}

bool Matrix::operator==(const Matrix& rMat) const
{
	if (rMat._cols != _cols || rMat._rows != _rows)
		return false;

	return values == rMat.values;
}
