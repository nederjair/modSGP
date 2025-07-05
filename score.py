import numpy as np

from control import get_control
from small_variation import applySVMat


def x0_score_calculation(
    x0,
    samples,
    x_count,
    model,
    h,
    int_method,
    modified_basic_solution,
    q_count,
    colCount,
    q,
    xf,
    u_min,
    u_max,
    u_count,
    min_delta,
):
    xi_minus_1 = x0
    ui_minus_1 = get_control(
        modified_basic_solution,
        x_count,
        q_count,
        colCount,
        q,
        xi_minus_1,
        xf,
        u_min,
        u_max,
        u_count,
    )

    # dis_to_xf = np.sqrt(np.sum((xi_minus_1[0:2]) ** 2))
    dis_to_xf = max(np.abs(xi_minus_1[0]), np.abs(xi_minus_1[1]))
    i = 1
    while dis_to_xf > min_delta and i < samples:
        xi = int_method(xi_minus_1, ui_minus_1, h, model)
        # dis_to_xf = np.sqrt(np.sum((xi[0:2]) ** 2))
        dis_to_xf = max(np.abs(xi[0]), np.abs(xi[1]))
        xi_minus_1 = xi
        ui_minus_1 = get_control(
            modified_basic_solution,
            x_count,
            q_count,
            colCount,
            q,
            xi_minus_1,
            xf,
            u_min,
            u_max,
            u_count,
        )
        i += 1

    if (
        i < samples
    ):  # means dis_to_xf is good and tf< t_max the robot reached the desired point
        i_f = i
        ui_minus_1 = np.zeros(u_count)
        while i < samples:
            xi = int_method(xi_minus_1, ui_minus_1, h, model)
            # dis_to_xf = np.sqrt(np.sum((xi[0:2]) ** 2))
            dis_to_xf = max(np.abs(xi[0]), np.abs(xi[1]))
            xi_minus_1 = xi
            i += 1
    else:
        i_f = i
    score = i_f * h + dis_to_xf
    return i_f, dis_to_xf, score


def x0_vec_score_calculation(
    x0_vec,
    samples,
    x_count,
    model,
    h,
    int_method,
    modified_basic_solution,
    q_count,
    colCount,
    q,
    xf,
    u_min,
    u_max,
    u_count,
    min_delta,
):
    scores = np.full(len(x0_vec), np.inf)
    i_f = np.full(len(x0_vec), np.inf)
    dis_to_xf = np.full(len(x0_vec), np.inf)
    for i in range(len(x0_vec)):
        x0_i_f, x0_dis_to_xf, x0_score = x0_score_calculation(
            x0_vec[i],
            samples,
            x_count,
            model,
            h,
            int_method,
            modified_basic_solution,
            q_count,
            colCount,
            q,
            xf,
            u_min,
            u_max,
            u_count,
            min_delta,
        )
        scores[i] = x0_score
        i_f[i] = x0_i_f
        dis_to_xf[i] = x0_dis_to_xf
    sorted_indexes = np.argsort(scores)

    return (
        i_f[sorted_indexes[-1]],
        dis_to_xf[sorted_indexes[-1]],
        scores[sorted_indexes[-1]],
    )


def group_score_calculation(
    x0_vec,
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
    group_size,
    sv_group,
    q_group,
    svRowCount,
    basic_solution,
):
    score = np.full(group_size, np.inf)
    i_f = np.full(group_size, np.inf)
    dis_to_xf = np.full(group_size, np.inf)
    for i in range(group_size):
        # basic solution applying (get the parse matrix representation of the current solution)
        modified_basic_solution = applySVMat(basic_solution, sv_group[i], svRowCount)
        # score calculation
        i_f[i], dis_to_xf[i], score[i] = x0_vec_score_calculation(
            x0_vec,
            samples,
            x_count,
            model,
            h,
            int_method,
            modified_basic_solution,
            q_count,
            colCount,
            q_group[i],
            xf,
            u_min,
            u_max,
            u_count,
            min_delta,
        )

    return i_f, dis_to_xf, score
