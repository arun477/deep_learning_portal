from multiprocessing import Pool

def f(x):
     return x**2

def square_and_sum(x):
    result = 0
    for i in range(x):
        result += i * i
    return result

