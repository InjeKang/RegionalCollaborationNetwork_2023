import os
from os.path import isdir, join

def set_dir():
    current_dir = os.getcwd()
    input_dir = join(current_dir, "data", "input")
    if os.path.isdir(input_dir):
        pass
    else:
        os.makedirs(input_dir)

set_dir()
