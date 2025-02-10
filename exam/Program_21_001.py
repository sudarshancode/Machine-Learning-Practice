#Crete 4x4 matrix and find its inverse

import numpy as np
import matplotlib.pyplot as plt

#Plot function
def plot_matrix(matrix,title):
    
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()

a=np.random.randint(1,10,(4,4))

print(f"4x4 Matrix:{a}") # a is a 4x4 matrix

det_a=np.linalg.det(a)
print(f"Determinante:{det_a}") #determinante of a

if det_a ==0:
    print("Matrix is singular")
else:
    a_inv=np.linalg.inv(a)
    #print(f"Inverse Matrix:{a_inv}")
    a_add=a+a_inv
    a_sub=a-a_inv
    a_mul=np.dot(a,a_inv)

    plot_matrix(a,"Original Matrix")
    plot_matrix(a_add,"Addition Matrix")
    plot_matrix(a_sub,"Subtraction Matrix")
    plot_matrix(a_mul,"Multiplication Matrix")



