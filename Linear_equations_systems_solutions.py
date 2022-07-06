import numpy as np

# S.T - транспонированная матрица S

# Метод простой итерации
def Simple_iteration_method(A, B, error):
	norm = 0 # норма матрицы A: квадратный корень суммы квадратов элементов матрицы
	n = A[0].size
	
	# вычисление нормы матрицы
	for i in range(n):
		for j in range(n):
			norm += A[i][j] ** 2 # возведение в квадрат
	norm **= 0.5 # взятие квадратного когня
	
	if (norm < 1):
		print('Norm of the matrix A: ', norm, '< 1. The method converges.')
	else:
		print('Norm of the matrix A: ', norm, '>= 1. The method don\'t converge.')
		return 0
	
	# проведение итераций
	count = 0 # счётчик итераций
	result = np.zeros(n) # решение системы линейных уравнений вида x=A*x+B
	for i in range(n):
		result[i] = B[i] # начальное приближение задаётся равным свободному вектору B
	max_coordinate_difference = error + 1 # максимальная разница предыдущего и следующего координат
	while (max_coordinate_difference > error):
		temp = np.zeros(n) # результат предыдущей итерации
		for i in range(n):
			temp[i] = result[i] # копирование результата предыдущей итерации
		
		# выполнение итерации
		for i in range(n):
			result[i] = B[i]
			for j in range(n):
				result[i] += A[i][j] * temp[j]
				
		# вычисление максимальной разницы предыдущего и следующего координат
		max_coordinate_difference = 0
		for i in range(n):
			if (abs(result[i] - temp[i]) > max_coordinate_difference):
				max_coordinate_difference = abs(result[i] - temp[i])
		
		count += 1
	
	# вывод количества итераций
	print('Number of iterations: ', count)
	
	return result

# Метод Зейделя
def Zeidel_method(A, B, error):
	norm = 0 # норма матрицы A: максимальная сумма модулей элементов строки матрицы
	n = A[0].size
	
	# вычисление нормы матрицы
	for i in range(n):
		temp = 0
		for j in range(n):
			temp += abs(A[i][j])
		if (temp > norm):
			norm = temp
	
	if (norm < 1):
		print('Norm of the matrix A: ', norm, '< 1. The method converges.')
	else:
		print('Norm of the matrix A: ', norm, '>= 1. The method don\'t converge.')
		return 0
	
	# проведение итераций
	count = 0 # счётчик итераций
	result = np.zeros(n, dtype=float) # решение системы линейных уравнений вида x=A*x+B
	for i in range(n):
		result[i] = B[i] # начальное приближение задаётся равным свободному вектору B
	max_coordinate_difference = error + 1 # максимальная разница предыдущего и следующего координат
	while (max_coordinate_difference > error):
		temp = np.zeros(n) # результат предыдущей итерации
		for i in range(n):
			temp[i] = result[i] # копирование результата предыдущей итерации
		
		# выполнение итерации
		for i in range(n):
			result[i] = B[i]
			for j in range(i):
				result[i] += A[i][j] * result[j]
			for j in range(i, n):
				result[i] += A[i][j] * temp[j]
		
		# вычисление максимальной разницы предыдущего и следующего координат
		max_coordinate_difference = 0
		for i in range(n):
			if (abs(result[i] - temp[i]) > max_coordinate_difference):
				max_coordinate_difference = abs(result[i] - temp[i])
		
		count += 1
	
	# вывод количества итераций
	print('Number of iterations: ', count)
	
	return result

# Метод квадратного корня
def Square_root_method(A, B, error = 0.5e-4):
	n = A[0].size
	S = np.zeros((n, n), dtype = complex) # верхняя треугольная матрица, элементы которой принимают комплексные значения (A=S.T*S)
	# формирование верхней треугольной матрицы S (A=S.T*S)
	for i in range(n):
		S[i][i] = A[i][i]
		for j in range(i):
			S[i][i] -= S[j][i] ** 2 # возведение в квадрат
		S[i][i] **= 0.5 # взятие квадратного когня
		for j in range(i + 1, n):
			S[i][j] = A[i][j]
			for k in range(i):
				S[i][j] -= S[k][i] * S[k][j]
			S[i][j] /= S[i][i]
	#print(S)
	
	# нахождение решения системы линейных уравнений вида: S.T*y=B
	solution_y = np.zeros(n, dtype = complex) # решение системы линейных уравнений S.T*y=B
	for i in range(n):
		temp = B[i]
		for j in range(i):
			temp -= S[j][i] * solution_y[j]
		solution_y[i] = temp / S[i][i]
	
	# нахождение решения системы линейных уравнений вида: S*x=y
	index = n - 1
	result = np.zeros(n, dtype = complex) # решение системы линейных уравнений S*x=y
	result[index] = solution_y[index] / S[index][index]
	for i in range(2, n + 1):
		index = n - i
		temp = solution_y[index]
		for j in range(index + 1, n):
			temp -= S[index][j] * result[j]
		result[index] = temp / S[index][index]
	
	# отбрасывание комплексной части, если она меньше погрешности error
	has_complex = False
	for i in range(n):
		if (result[i].imag < error): # сравнение мнимой части
			result[i] = result[i].real # действительная часть
		else:
			has_complex = True
	if (has_complex == False):
		temp = np.zeros(n)
		for i in range(n):
			temp[i] = result[i].real # действительная часть
		result = temp
	return result

# функция вывода решения системы линейных уравнений
def print_solution(solution):
	for i in range(solution.size):
		print('\t', i + 1, ': ', solution[i])
	print()

def main():
	error = 0.5e-4 # погрешность
	A = np.array([[0.23, -0.04, 0.21, -0.18], [0.45, -0.23, 0.06, 0], [0.26, 0.34, -0.11, 0], [0.05, -0.26, 0.34, -0.12]]) # исходная матрица системы линейных уравнений для методов простой итерации и Зейделя
	B = np.array([1.24, -0.88, 0.62, -1.17]) # исходный свободный вектор линейных уравнений для методов простой итерации и Зейделя
	
	# вызов методов простой итерации и Зейделя
	print('Simple iteration method:')
	print_solution(Simple_iteration_method(A, B, error))
	print('Zeidel method:')
	print_solution(Zeidel_method(A, B, error))
	
	A = np.array([[3.14, -2.12, 1.17], [-2.12, 1.32, -2.45], [1.17, -2.45, 1.18]]) # исходная матрица системы линейных уравнений для метода квадратного корня
	B = np.array([1.27, 2.13, 3.14]) # исходный свободный вектор линейных уравнений для метода квадратного корня
	
	# вызов метода квадратного корня
	print('Square root method:')
	print_solution(Square_root_method(A, B))

if __name__ == '__main__':
    main()
