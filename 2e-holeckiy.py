# Розклад Холецького — це алгоритм факторизації матриці, який розкладає симетричну додатно визначену матрицю 
    # на добуток нижньої трикутної матриці та її транспонування. (спрощення обчислень і оптимізації часу)



from numpy import array
from numpy.linalg import cholesky
# define a 3x3 matrix
A = array([[36, 30, 18], [30, 41, 23], [18, 23, 14]])
print(A)
# Cholesky decomposition
L = cholesky(A)
print(L)
print(L.T)
# reconstruct
B = L.dot(L.T)
print(B)