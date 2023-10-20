from time import sleep
import timeit
def f():
    sleep(1)


if __name__ == '__main__':
    execution_time = timeit.timeit(f, number=5)
    print(f"{execution_time:.6f} seconds")

