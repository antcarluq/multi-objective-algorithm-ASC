import numpy as numpy
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from main import *
from subproblem import Subproblem
from subproblem import Individual

import os

result = os.popen("metrics/metrics " + "./metrics/inputfiles/ZDT3/EVAL4000/P40G100/seed1.in").read()
print(result)