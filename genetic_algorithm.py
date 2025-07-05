from random import randrange, uniform

import numpy as np

from score import group_score_calculation


# selection algorithm
def roulette(normalized_probabilities):
    r = uniform(0, 1)
    f = normalized_probabilities[0]
    k = 0
    while f < r:
        k += 1
        f += normalized_probabilities[k]
    return k


# probability calculation methods
def prob_inverse(scores):
    inv_prob = np.full_like(scores, 0.0)
    score_sum = np.sum(scores)
    for k in range(len(scores)):
        if scores[k] < 1.0e-9:
            inv_prob[k] = score_sum / 1.0e-9
        else:
            inv_prob[k] = score_sum / scores[k]
    inv_prob_sum = np.sum(inv_prob)
    norm_inv_prob = np.true_divide(inv_prob, inv_prob_sum)
    return norm_inv_prob


def prob_bias(score):
    sorted_indexes = np.argsort(score)
    max_index = sorted_indexes[-1]
    max_score = score[max_index]
    biased_score = 1.2 * max_score - score
    return np.true_divide(biased_score, np.sum(biased_score))


def sum_based_prob(scores):
    score_sum = np.sum(scores)
    biased_score = score_sum - scores
    return np.true_divide(biased_score, np.sum(biased_score))


def parents_index_selection(probabilities):
    parent_index_1 = roulette(probabilities)
    parent_index_2 = roulette(probabilities)
    while parent_index_1 == parent_index_2:
        parent_index_2 = roulette(probabilities)
    return parent_index_1, parent_index_2


def crossover_mat_sv(parent_1, parent_2):
    sv_mat_row_count1, sv_mat_col_count1 = parent_1.shape
    sv_mat_row_count2, sv_mat_col_count2 = parent_2.shape
    assert sv_mat_row_count1 == sv_mat_row_count2
    child1 = parent_1.copy()
    child2 = parent_2.copy()
    if sv_mat_row_count1 > 1:
        ia = randrange(1, sv_mat_row_count1)
        ib = randrange(1, sv_mat_row_count1)
        i1 = min([ia, ib])
        i2 = max([ia, ib])
        for elemIndex in range(i1, i2 + 1):
            temp = child1[elemIndex].copy()
            child1[elemIndex] = child2[elemIndex].copy()
            child2[elemIndex] = temp
        return child1, child2
    else:
        return child1, child2


def crossover_q(q1, q2):
    (q_count1,) = q1.shape
    (q_count2,) = q2.shape
    assert q_count1 == q_count2
    child1 = q1.copy()
    child2 = q2.copy()
    if q_count1 > 1:
        ia = randrange(1, q_count1)
        ib = randrange(1, q_count1)
        i1 = min([ia, ib])
        i2 = max([ia, ib])
        for elemIndex in range(i1, i2 + 1):
            temp = child1[elemIndex].copy()
            child1[elemIndex] = child2[elemIndex].copy()
            child2[elemIndex] = temp
        return child1, child2
    return child1, child2


def crossover(parent_1, parent_2, parent_q_1, parent_q_2):
    child_sv_1, child_sv_2 = crossover_mat_sv(parent_1, parent_2)
    child_q_1, child_q_2 = crossover_q(parent_q_1, parent_q_2)
    offsprings_sv = np.zeros((16, parent_1.shape[0], parent_1.shape[1]), dtype=int)
    offsprings_sv[0:4] = parent_1.copy()
    offsprings_sv[4:8] = parent_2.copy()
    offsprings_sv[8:12] = child_sv_1.copy()
    offsprings_sv[12:16] = child_sv_2.copy()
    offsprings_q = np.zeros((16, parent_q_1.shape[0]))
    offsprings_q[0] = parent_q_1.copy()
    offsprings_q[1] = parent_q_2.copy()
    offsprings_q[2] = child_q_1.copy()
    offsprings_q[3] = child_q_2.copy()
    offsprings_q[4] = parent_q_1.copy()
    offsprings_q[5] = parent_q_2.copy()
    offsprings_q[6] = child_q_1.copy()
    offsprings_q[7] = child_q_2.copy()
    offsprings_q[8] = parent_q_1.copy()
    offsprings_q[9] = parent_q_2.copy()
    offsprings_q[10] = child_q_1.copy()
    offsprings_q[11] = child_q_2.copy()
    offsprings_q[12] = parent_q_1.copy()
    offsprings_q[13] = parent_q_2.copy()
    offsprings_q[14] = child_q_1.copy()
    offsprings_q[15] = child_q_2.copy()
    return offsprings_sv, offsprings_q


def crossover_cycle(
    pop_sv,
    pop_q,
    samples,
    probabilities,
    cross_return_count,
    cross_pop,
    sv_mat_row_count,
    q_count,
    x0,
    x_count,
    model,
    h,
    int_method,
    colCount,
    xf,
    u_min,
    u_max,
    u_count,
    min_delta,
    sv_row_count,
    basic_solution,
):
    pop_sv_new = np.zeros((cross_pop, sv_mat_row_count, 4), dtype=int)
    pop_q_new = np.zeros((cross_pop, q_count))
    pop_scores_new = np.full(cross_pop, np.inf)
    k = 0
    while k < cross_pop:
        parent_index_1, parent_index_2 = parents_index_selection(probabilities)
        parent_1 = pop_sv[parent_index_1]
        parent_2 = pop_sv[parent_index_2]
        parent_q_1 = pop_q[parent_index_1]
        parent_q_2 = pop_q[parent_index_2]
        sv_offspring_size = 16
        offsprings_sv, offsprings_q = crossover(
            parent_1, parent_2, parent_q_1, parent_q_2
        )
        _, _, offsprings_scores = group_score_calculation(
            x0,
            samples,
            x_count,
            model,
            h,
            int_method,
            q_count,
            colCount,
            xf,
            u_min,
            u_max,
            u_count,
            min_delta,
            sv_offspring_size,
            offsprings_sv,
            offsprings_q,
            sv_row_count,
            basic_solution,
        )
        if cross_return_count == 16:
            pop_sv_new[k : k + cross_return_count] = offsprings_sv[:cross_return_count]
            pop_q_new[k : k + cross_return_count] = offsprings_q[:cross_return_count]
            pop_scores_new[k : k + cross_return_count] = offsprings_scores[
                :cross_return_count
            ]
        else:
            offsprings_sorted_indexes = np.argsort(offsprings_scores)
            pop_sv_new[k : k + cross_return_count] = offsprings_sv[
                offsprings_sorted_indexes[:cross_return_count]
            ]
            pop_q_new[k : k + cross_return_count] = offsprings_q[
                offsprings_sorted_indexes[:cross_return_count]
            ]
            pop_scores_new[k : k + cross_return_count] = offsprings_scores[
                offsprings_sorted_indexes[:cross_return_count]
            ]
        k += cross_return_count
    return pop_sv_new, pop_q_new, pop_scores_new
