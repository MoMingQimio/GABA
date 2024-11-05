
from train_for_main import train
import multiprocessing as mp
from multiprocessing import Pool, Process


def main():
    ra = [i for i in range(3)]
    proces = []
    for i in ra:
        proc = Process(target=train, args=(str(i + 1),))
        proces.append(proc)
        proc.start()

    # complete
    for proc in proces:
        proc.join()


if __name__ == '__main__':
    main()