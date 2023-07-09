# -*- coding: utf-8 -*-
import os


def exists(problem_file=None):
    """
    Check if the problem file exists.
    The function returns None if the file does not exist or the argument is None.
    """
    if problem_file is not None:
        return os.path.isfile(problem_file)

    return False
