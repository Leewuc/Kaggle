import math
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import itertools
import functools
import time
import csv
import os
import sys
import traceback
NV = 15
MAX_N = 200
PI = math.pi

SA_T0 = 2.5
SA_TM = 5e-7
SA_ITER_BASE = 15000
SA_RESTART_BASE = 16

LOCAL_SEARCH_ITERS = 150
COMPACTION_ITERS = 80
EDGE_SLIDE_ITERS = 12

REMOVE_RATIO = 0.50
REINSERT_TRY = 200

BACKPROP_MAX_PASS = 10
BACKPROP_RANGE = 5

EPS = 1e-12
BBOX_EPS = 0.01

VERBOSE = True
DEBUG = False

