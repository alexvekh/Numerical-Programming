# 1. Обчислення добутку AA T Якщо матриця A прямокутна, ця операція дозволить отримати квадратну матрицю. Це важливо для подальших операцій.
import numpy as np

# Приклад матриці
# matrix_a = np.array([[1, 2, 3],
#                     [4, 5, 6]])

matrix_a = np.array([[4, 0],
                    [3, -5]])

# Обчислення транспонованої матриці
matrix_a_transposed = np.transpose(matrix_a)  # або matrix_a.T

# Обчислення добутку матриці на її транспоновану
result = np.dot(matrix_a_transposed, matrix_a)  # або matrix_a @ matrix_a_transposed

# Виведення результату
print("Матриця A:")
print(matrix_a)
print("\nТранспонована матриця A:")
print(matrix_a_transposed)
print("\nДобуток матриці A на її транспоновану:")
print(result)
 # Матриця A: 				[[ 4 0] [ 3 -5]]
 # Транспонована матриця A: 	[[ 4 3] [ 0 -5]]
 # Добуток матриці A на її транспоновану:	[[ 25 -15] [-15 25]]


# 2. Обчислення власних значень та векторів.
	#Для цього треба записати характеристичне рівняння та розв'язати його. Детально ця процедура була розглянута у попередній темі.
# Обчислення власних векторів та власних значень
eigenvalues, eigenvectors = np.linalg.eig(result)
# Виведення результату
print("Власні значення:")
print(eigenvalues)
print("\nВласні вектори:")
print(eigenvectors)
# Власні значення: [40. 10.]
# Власні вектори:  [[ 0.70710678 0.70710678] [-0.70710678 0.70710678]]

# 3. Тепер, коли ми отримали власні вектори та власні значення, необхідно скласти матриці V та Σ.
# Отримання індексів, які відсортовують масив за першим стовпцем
sorted_indices = np.argsort(eigenvectors[:, 0])
V = eigenvectors[sorted_indices]

# Виведення результату
print("Вихідний масив:")
print(eigenvectors)
print("\nМатриця V:")
print(V)
# Вихідний масив: 	[[ 0.70710678 0.70710678]  [-0.70710678 0.70710678]]
# Матриця V: 		[[-0.70710678 0.70710678]  [ 0.70710678 0.70710678]]

# діагональні значення в сигмах завжди розташовані в порядку спадання, тому вектори також розміщуються у відповідному порядку.
# Створення діагонального масиву
Sigma = np.diag(sorted(eigenvalues, reverse=True))
Sigma = np.sqrt(Sigma)
print("\nМатриця Σ:")
print(Sigma)
# Матриця Σ: [[6.32455532 0.    ]  [0.     3.16227766]]
Sigma.transpose()
# array([[6.32455532, 0.    ],     [0.    , 3.16227766]])
np.array([[1,2],
          [3,4]]).T
# array([[1, 3],    [2, 4]])

#  головні компоненти відповідають верхньому k-му діагональному елементу, який фіксує найбільшу дисперсію. Чим вище значення, тим важливіший компонент і тим більшу дисперсію він описує.

# 4 Тепер, коли у нас є матриці V і сигма, настав час знайти U.
# 		A = UΣV 		AVΣ = U

AV = np.dot(matrix_a, V)

# Далі нам потрібно перетворити матрицю так, щоб отримати на одиничні вектори у стовпцях матриці.
# Спосіб, яким ми робимо це, беремо значення в стовпцях і ділимо їх на квадратний корінь із суми квадратів значень. 
# Отже, у цьому випадку ми робимо наступне:
def matrix_norm(mtr):
    # Обчислення кореня квадратного з суми елементів для кожного стовпця
    sqrt_sum_columns = np.sqrt(np.sum(mtr**2, axis=0))
    res = mtr / sqrt_sum_columns
    return res
AV = matrix_norm(AV)
print(AV)
	# [[-0.4472136  0.89442719]  [-0.89442719 -0.4472136 ]]
U = np.dot(AV, Sigma.T)
U = matrix_norm(U)
print('Матриця U')
print(U)
	# Матриця U [[-0.4472136  0.89442719]  [-0.89442719 -0.4472136 ]]
    

# Отже, ми обчислили U, сигму та V і розклали матрицю A на три матриці, як наведено нижче.

A = np.dot(np.dot(U, Sigma), V.T)
print('Початкова матриця A')
print(A)

	# Початкова матриця A [[ 4.00000000e+00 2.73432346e-16]  [ 3.00000000e+00 -5.00000000e+00]]
# Порівняємо результат з розкладом за допомогою функції
print(U)
print(Sigma)
print(V)
	# [[-0.4472136  0.89442719] [-0.89442719 -0.4472136 ]]
	# [[6.32455532 0.    ]  [0.     3.16227766]]
	# [[-0.70710678 0.70710678]  [ 0.70710678 0.70710678]]

Uu, Ss, Vh = np.linalg.svd(matrix_a)
print('SVD U ')
print(Uu)
print('SVD Sigma ')
print(Ss)
print('SVD Vh ')
print(Vh)
	# SVD U  [[-0.4472136 -0.89442719]  [-0.89442719 0.4472136 ]]
	# SVD Sigma  [6.32455532 3.16227766]
	# SVD Vh 	[[-0.70710678 0.70710678] [-0.70710678 -0.70710678]]

# Отже, можемо бачити, що результати в цілому співпадають з точністю до знаків деяких векторів