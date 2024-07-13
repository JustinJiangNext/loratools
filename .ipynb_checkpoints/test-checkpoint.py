import multiprocessing

from multiprocessing import Process, Manager

with Manager() as manager:
    print('hello')