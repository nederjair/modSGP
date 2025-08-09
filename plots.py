import matplotlib.pyplot as plt

from sgp import symDecodeMatrix
from small_variation import applySVMat
from states import trajectory_calculation, trajectory_calculation_with_derivatives


def plot_x0_vec_trajectories(
    sv_mat,
    xCount,
    q_count,
    colCount,
    x0_vec,
    samples,
    h,
    model,
    int_method,
    q,
    xf,
    u_min,
    u_max,
    uCount,
    min_delta,
    svRowCount,
    basic_solution,
):
    # ----------------control expression calculation-----------------------------------------#
    # basic solution applying (get the parse matrix representation of the current solution)
    modified_basic_solution = applySVMat(basic_solution, sv_mat, svRowCount)
    expression_u1 = symDecodeMatrix(
        modified_basic_solution[0], xCount, q_count, colCount
    )
    expression_u2 = symDecodeMatrix(
        modified_basic_solution[1], xCount, q_count, colCount
    )
    print("---------------------------------------------")
    print("solution:\n", modified_basic_solution)
    print("svMat:\n", sv_mat)
    print("q: ", q)
    print("u1 = ", expression_u1)
    print("u2 = ", expression_u2)
    for x0_index in range(len(x0_vec)):
        x, u, i_f, dis_to_xf, score = trajectory_calculation(
            x0_vec[x0_index],
            samples,
            xCount,
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
            uCount,
            min_delta,
        )
        print("////////////////////////////////////////")
        print("x0:", x0_vec[x0_index])
        print("t_max = ", samples * h)
        print("tf = ", i_f * h)
        print("dis to xf = ", dis_to_xf)
        print("score = ", score)
        print("////////////////////////////////////////")

        plt.plot(x0_vec[x0_index][0], x0_vec[x0_index][1], "o")
        plt.plot(x[0], x[1], "-")
    plt.plot(0, 0, "o")
    plt.show()
    print("---------------------------------------------")


def plot_others(
    sv_mat,
    xCount,
    q_count,
    colCount,
    x0_vec,
    samples,
    h,
    model,
    int_method,
    q,
    xf,
    u_min,
    u_max,
    uCount,
    min_delta,
    svRowCount,
    basic_solution,
):
    # ----------------control expression calculation-----------------------------------------#
    # basic solution applying (get the parse matrix representation of the current solution)
    modified_basic_solution = applySVMat(basic_solution, sv_mat, svRowCount)
    expression_u1 = symDecodeMatrix(
        modified_basic_solution[0], xCount, q_count, colCount
    )
    expression_u2 = symDecodeMatrix(
        modified_basic_solution[1], xCount, q_count, colCount
    )
    t = [ti * h for ti in range(0, samples, 1)]
    while True:
        cont = input(
            "if you want to continue press enter,\nif you want to stop type the letter n and then press enter: "
        )
        if cont == "n":
            break
        x0_indexes = input("enter the x0s you want to plot: ")
        x0_indexes = x0_indexes.split(" ")
        for x0_index in x0_indexes:
            x, dx_dt, u, i_f, dis_to_xf, score = (
                trajectory_calculation_with_derivatives(
                    x0_vec[int(x0_index)],
                    samples,
                    xCount,
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
                    uCount,
                    min_delta,
                )
            )
            plt.figure(1)
            plt.plot(t, x[0], "-", label="x0" + str(int(x0_index)))
            plt.plot(x0_vec[int(x0_index)][0], "o")
            plt.legend()
            plt.xlabel("time")
            plt.ylabel("x1")

            plt.figure(2)
            plt.plot(t, x[1], "-", label="x0" + str(int(x0_index)))
            plt.plot(x0_vec[int(x0_index)][1], "o")
            plt.legend()
            plt.xlabel("time")
            plt.ylabel("x2")

            plt.figure(3)
            plt.plot(t, x[2], "-", label="x0" + str(int(x0_index)))
            plt.plot(x0_vec[int(x0_index)][2], "o")
            plt.legend()
            plt.xlabel("time")
            plt.ylabel("x3")

            plt.figure(4)
            plt.plot(t, dx_dt[0], "-", label="x0" + str(int(x0_index)))
            plt.legend()
            plt.xlabel("time")
            plt.ylabel("dx1/dt")

            plt.figure(5)
            plt.plot(t, dx_dt[1], "-", label="x0" + str(int(x0_index)))
            plt.legend()
            plt.xlabel("time")
            plt.ylabel("dx2/dt")

            plt.figure(6)
            plt.plot(t, dx_dt[2], "-", label="x0" + str(int(x0_index)))
            plt.legend()
            plt.xlabel("time")
            plt.ylabel("dx3/dt")

            plt.figure(7)
            plt.plot(t, u[0], "-", label="x0" + str(int(x0_index)))
            plt.legend()
            plt.xlabel("time")
            plt.ylabel("u1")

            plt.figure(8)
            plt.plot(t, u[1], "-", label="x0" + str(int(x0_index)))
            plt.legend()
            plt.xlabel("time")
            plt.ylabel("u2")

        plt.show()
