import numpy as np
from random import uniform


def encode_q(q_count, q_min, q_max):
    q_vec = np.zeros(q_count)
    for k in range(q_count):
        q_vec[k] = uniform(0, 1) * (q_max - q_min) + q_min
    return q_vec
