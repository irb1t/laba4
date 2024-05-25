# laba4
30.	Формируется матрица F следующим образом: скопировать в нее А и  если в В количество простых чисел в нечетных столбцах,
чем произведение чисел по периметру С, то поменять местами  С и В симметрично, иначе С и В поменять местами несимметрично.
	При этом матрица А не меняется. После чего если определитель матрицы А больше суммы диагональных элементов матрицы F, то
вычисляется выражение: A*AT – K * F, иначе вычисляется выражение (A-1 +G-F-1)*K, где G-нижняя треугольная матрица, полученная из А.
Выводятся по мере формирования А, F и все матричные операции последовательно.
Для ИСТд-13
D	Е
С	В
С клавиатуры вводится два числа K и N. Квадратная матрица А(N,N), состоящая из 4-х равных по размерам подматриц,
 B,C,D,E заполняется случайным образом целыми числами в интервале [-10,10].
 Для отладки использовать не случайное заполнение, а целенаправленное. Вид матрицы А: 


import numpy as np
import matplotlib.pyplot as plt

def create_matrix(N, seed=0):
    np.random.seed(seed)
    return np.random.randint(-10, 11, size=(N, N))

def split_matrix(A):
    n = A.shape[0] // 2
    C = A[:n, :n]
    E = A[:n, n:]
    D = A[n:, :n]
    B = A[n:, n:]
    return B, C, D, E

def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(np.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

def count_primes_in_odd_columns(matrix):
    count = 0
    for col in range(0, matrix.shape[1], 2):
        count += np.sum([is_prime(num) for num in matrix[:, col]])
    return count

def product_perimeter(matrix):
    top = np.prod(matrix[0, :])
    bottom = np.prod(matrix[-1, :])
    left = np.prod(matrix[:, 0])
    right = np.prod(matrix[:, -1])
    return top * bottom * left * right

def form_matrix_F(A, B, C, D, E):
    n = A.shape[0] // 2
    F = np.copy(A)
    F[:n, :n] = C
    F[:n, n:] = E
    F[n:, :n] = D
    F[n:, n:] = B
    return F

# Ввод данных
N = int(input("Введите N (размер матрицы): "))
K = int(input("Введите K: "))

# Создание и вывод матрицы A
A = create_matrix(N)
print("Matrix A:")
print(A)

# Разделение на подматрицы
B, C, D, E = split_matrix(A)
print("Submatrix B:")
print(B)
print("Submatrix C:")
print(C)
print("Submatrix D:")
print(D)
print("Submatrix E:")
print(E)

# Проверка условий и замена подматриц
prime_count_B = count_primes_in_odd_columns(B)
perimeter_product_C = product_perimeter(C)

print("Prime count in B:", prime_count_B)
print("Perimeter product of C:", perimeter_product_C)

if prime_count_B > perimeter_product_C:
    C, B = np.copy(B), np.copy(C)
else:
    C, B = np.copy(B.T), np.copy(C.T)

print("Updated Submatrix B:")
print(B)
print("Updated Submatrix C:")
print(C)

# Формирование матрицы F
F = form_matrix_F(A, B, C, D, E)
print("Matrix F:")
print(F)

# Вычисление выражений
det_A = np.linalg.det(A)
sum_diag_F = np.trace(F)

print("Determinant of A:", det_A)
print("Sum of diagonal elements of F:", sum_diag_F)

if det_A > sum_diag_F:
    result = np.dot(A, A.T) - K * F
else:
    G = np.tril(A)
    result = (np.linalg.inv(A) + G - np.linalg.inv(F)) * K

print("Result:")
print(result)

# Визуализация результатов
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.title("Matrix A")
plt.imshow(A, cmap='viridis', interpolation='none')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.title("Matrix F")
plt.imshow(F, cmap='viridis', interpolation='none')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.title("Result Matrix")
plt.imshow(result, cmap='viridis', interpolation='none')
plt.colorbar()

plt.tight_layout()
plt.show()
