import numpy as np

from control import get_control


def trajectory_calculation(
    x0,
    samples,
    x_count,
    model,
    h,
    int_method,
    mat_set,
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
        mat_set, x_count, q_count, colCount, q, xi_minus_1, xf, u_min, u_max, u_count
    )
    x = np.zeros((x_count, samples))
    u = np.zeros((u_count, samples))
    u[:, 0] = ui_minus_1
    x[:, 0] = xi_minus_1
    dis_to_xf = max(np.abs(xi_minus_1[0]), np.abs(xi_minus_1[1]))
    # dis_to_xf = np.sqrt(np.sum((xi_minus_1[0:2]) ** 2))
    i = 0
    while dis_to_xf > min_delta and i < samples - 1:
        i += 1
        x[:, i] = int_method(xi_minus_1, ui_minus_1, h, model)
        # dis_to_xf = np.sqrt(np.sum((x[0:2, i]) ** 2))
        dis_to_xf = max(np.abs(x[0, i]), np.abs(x[1, i]))
        xi_minus_1 = x[:, i]
        ui_minus_1 = get_control(
            mat_set,
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
        u[:, i] = ui_minus_1

    if (
        i < samples - 1
    ):  # means dis_to_xf is good and tf< t_max the robot reached the desired point
        i_f = i
        u[:, i:] = 0.0
        ui_minus_1 = u[:, i]
        while i < samples - 1:
            i += 1
            x[:, i] = int_method(xi_minus_1, ui_minus_1, h, model)
            # dis_to_xf = np.sqrt(np.sum((x[0:2, i]) ** 2))
            dis_to_xf = max(np.abs(x[0, i]), np.abs(x[1, i]))
            xi_minus_1 = x[:, i]
            ui_minus_1 = u[:, i]
    else:
        i_f = i + 1
    score = i_f * h + dis_to_xf
    return x, u, i_f, dis_to_xf, score


def trajectory_calculation_with_derivatives(
    x0,
    samples,
    x_count,
    model,
    h,
    int_method,
    mat_set,
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
        mat_set, x_count, q_count, colCount, q, xi_minus_1, xf, u_min, u_max, u_count
    )
    x = np.zeros((x_count, samples))
    dx_dt = np.zeros((x_count, samples))
    u = np.zeros((u_count, samples))
    u[:, 0] = ui_minus_1
    x[:, 0] = xi_minus_1
    dis_to_xf = max(np.abs(xi_minus_1[0]), np.abs(xi_minus_1[1]))
    # dis_to_xf = np.sqrt(np.sum((xi_minus_1[0:2]) ** 2))
    i = 0
    while dis_to_xf > min_delta and i < samples - 1:
        i += 1
        x[:, i] = int_method(xi_minus_1, ui_minus_1, h, model)
        dx_dt[:, i] = model(xi_minus_1, ui_minus_1)
        # dis_to_xf = np.sqrt(np.sum((x[0:2, i]) ** 2))
        dis_to_xf = max(np.abs(x[0, i]), np.abs(x[1, i]))
        xi_minus_1 = x[:, i]
        ui_minus_1 = get_control(
            mat_set,
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
        u[:, i] = ui_minus_1

    if (
        i < samples - 1
    ):  # means dis_to_xf is good and tf< t_max the robot reached the desired point
        i_f = i
        u[:, i:] = 0.0
        ui_minus_1 = u[:, i]
        while i < samples - 1:
            i += 1
            x[:, i] = int_method(xi_minus_1, ui_minus_1, h, model)
            dx_dt[:, i] = model(xi_minus_1, ui_minus_1)
            # dis_to_xf = np.sqrt(np.sum((x[0:2, i]) ** 2))
            dis_to_xf = max(np.abs(x[0, i]), np.abs(x[1, i]))
            xi_minus_1 = x[:, i]
            ui_minus_1 = u[:, i]
    else:
        i_f = i + 1
    score = i_f * h + dis_to_xf
    return x, dx_dt, u, i_f, dis_to_xf, score
