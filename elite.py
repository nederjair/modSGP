def stack_elite(sv_mat, q, score, elite_svs, elite_qs, elite_scores, elite_count):
    if score in elite_scores:
        return elite_svs, elite_qs, elite_scores
    else:
        j = elite_count - 1
        while score < elite_scores[j] and j >= 0:
            j -= 1
        if j < elite_count - 1:
            elite_svs_temp = elite_svs[j + 1:-1].copy()
            elite_qs_temp = elite_qs[j + 1:-1].copy()
            elite_scores_temp = elite_scores[j + 1:-1].copy()

            elite_svs[j + 1] = sv_mat.copy()
            elite_qs[j + 1] = q.copy()
            elite_scores[j + 1] = score

            elite_svs[j + 2:] = elite_svs_temp
            elite_qs[j + 2:] = elite_qs_temp
            elite_scores[j + 2:] = elite_scores_temp
    return elite_svs, elite_qs, elite_scores


def stack_group_elite(sorted_sv_group, sorted_q_group, sorted_score_group, elite_svs, elite_qs, elite_scores,
                      elite_count, group_count):
    for i in range(group_count):
        if sorted_score_group[i] >= elite_scores[-1]:
            return elite_svs, elite_qs, elite_scores
        else:
            elite_svs, elite_qs, elite_scores = stack_elite(sorted_sv_group[i], sorted_q_group[i],
                                                            sorted_score_group[i], elite_svs, elite_qs, elite_scores,
                                                            elite_count)
    return elite_svs, elite_qs, elite_scores
