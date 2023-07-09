# -*- coding: utf-8 -*-
import pickle


def deserialize_problem(problem_file):
    with open(problem_file, "rb") as infile:
        return pickle.load(infile)


def serialize_problem(problem, problem_file):
    with open(problem_file, "wb") as outfile:
        pickle.dump(problem, outfile)
