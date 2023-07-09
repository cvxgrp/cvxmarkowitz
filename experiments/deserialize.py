# -*- coding: utf-8 -*-
import pickle

if __name__ == "__main__":
    with open("test.pickle", "rb") as infile:
        test = pickle.load(infile)
        print(type(test))
