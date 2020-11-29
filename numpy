"""
Numpy:
- stands for numeric python
- used for scientific computing and dat anaalysis
- basic data structure in numpy multidimensional array, ndarray.
- imported as "import numpy as np"
- ways to create numpy arrays:
1) using np.array()
2) using np.ones()
3) using np.zeros()
4) using np.random.random() in  [0.0, 1.0)
5) using np.arange()
6) using linspace(),
7) using full(),
8) using tile(),
9) using eye(),
10) using randint()
"""
import numpy as np
# creating 1d array
val = np.array([1,2,3,4,5])
print("using np.array(), 1d array: \n", val)
print("Type of resultant list: ", type(val))
# creating 2d array
list1 = [1,2,3]
list2 = [4,5,6]
list3 = [5,6,7]
val = np.array([list1, list2, list3])
print("using np.array(), 2d array: \n",val)
print(type(val))
print("using ones:(1d array) \n", np.ones(5,dtype=int))
print("using ones:(2d array) \n", np.ones((2,3),dtype=int))
print("using zeros:(1d array) \n", np.zeros(5,dtype=int))
print("using zeros:(2d array) \n", np.zeros((2,3), dtype=int))
print("using random.random:(1d array) \n", np.random.random(5))
print("using arange,", np.arange(3,9,1,float))
print("using linspace:", np.linspace(10,40,5))
print("using full", np.full(5, 9))
print("using tile", np.tile([0,1],(4,2)))
print("using eye", np.eye(2))
# very useful
print("using randint", np.random.randint(4,6,size=10))
"""
Problem:
    Checkerboard Matrix
Description
Given an even integer ‘n’, create an ‘n*n’ checkerboard matrix with the values 0 and 1, using the tile function.
 
Format:
Input: A single even integer 'n'.
Output: An 'n*n' NumPy array in checkerboard format.

Example:
Input 1:
2
Output 1:
[[0 1]
 [1 0]]
Input 2:
4
Output 2:
[[0 1 0 1] 
 [1 0 1 0]
 [0 1 0 1]
 [1 0 1 0]]
# Read the variable from STDIN

n = int(input())
print(np.tile([[0,1],[1,0]],(n//2,n//2)))
"""
#---------------------------------------------------------------------------
print(np.arange(24).reshape(2,3,4))
"""Multi-dimensional arrays"""
# creating a random array of 4 rows and 3 columns
rand_array = np.random.random((4,3))
print("Using random, creating a random array of 20 rows & 10 columns: \n")
print("printing entire array: ", rand_array)
print("printing first row: ", rand_array[1])
"""Accessing 1d numpy array is same as of lists, but not for multi-dimensional arrays
to access 2nd element of third list in array of lists - List[2][1]
to access 2nd element of third numpy-array in array of lists - Array[2,1]
All other conventions remain same - slicing [2:3],[],[2: ],[ :3]
array[:,:2] - all rows, but only till first 2 columns in numpy array
"""
"""numpy Array Dimensions"""
print(rand_array.ndim)
print(rand_array.shape)
# --------------------------------------------------------------------------
# get numpy array with start and end along with fixed gap and data type:
print(np.arange(2,20,2))
# reshape this result into 3 separate arrays
print(np.arange(2,20,2).reshape(3,3,1))
# --------------------------------------------------------------------------
"""Operations on numpy arrays:
1) Manipulate arrays:
- reshape()
- Transpose
- stack

1) reshape(): reshape the numpy array to specified dimensions(-1 for auto dimension)
syntax: reshape(no_of_partition, rows,columns)
NOTE: if no.of parameters = 2, -> Treated as row, col
      if no. of parameters = 3, -> Treated as no_of_partition, row, col

2) Transpose: np_array.T will give transpose of this array(matrix)

3) STacking: 2 types
a) horizontal stacking : 
Note: no. of rows in both arrays(matrices) must be same
syntax : np.hstack(np_array1, np_array2)
Note: Resultant array will have all rows of both arrays
b) vertical stacking : 
Note: no. of columns in both arrays(matrices) must be same
syntax : np.vstack(np_array1, np_array2)
Note: Resultant array will have all columns of both array
"""
def func(x):
    return x ** 2
a = np.arange(1,20,2)
f = np.vectorize(func)
print("After using Vectorise", f(a))
"""
Basic Linear algebra operaion on NUmpy arrays: below things are only for 2d arrays
1) Determinant - np.linalg.det(np_array)
2) Inverse - np.linalg.inv(np_array)
3) Eigen values and vectors - np.linalg.eig(np_array)
4) Multiplication - np.dot(np_array_1, np_array_2)
NOTE: np_array1 * np_array2 is different.
"""
