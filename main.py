import numpy as np

from coefficients import encode_q
from control import encode_u_simple
from elite import stack_group_elite
from genetic_algorithm import crossover_cycle, sum_based_prob
from integration import Methods
from model import TwoWheels
from plots import plot_x0_vec_trajectories
from score import group_score_calculation
from small_variation import applySVMat, encodeSVMat

# score calculation parameters
h = 0.1
samples = 25
# model parameters
xCount = 3
qCount = 6
uCount = 2
u_min = -10
u_max = 10
# model object
robot = TwoWheels(length=2, radius=1)
# initial states
X0 = 3
Y0 = 3.5
Z0 = np.pi
x00 = np.array([-X0, Y0, Z0 * 4 / 15])
x01 = np.array([-X0, -Y0, Z0 * 4 / 15])
x02 = np.array([X0, -Y0, Z0 * 4 / 15])
x03 = np.array([X0, Y0, Z0 * 4 / 15])
"""x04 = np.array([-X0,  Y0,  -Z0*4/15])
x05 = np.array([-X0, -Y0, -Z0*4/15])
x06 = np.array([X0,  -Y0,  -Z0*4/15])
x07 = np.array([X0,   Y0,  -Z0*4/15])"""
"""x0_vec = np.array([[-X0,  Y0,  Z0*0.0], [-X0/2,  Y0,  Z0*0.0], [-X0,  Y0/2,  Z0*0.0], [-X0/2,  Y0/2,  Z0*0.0],
                   [-X0,  -Y0,  Z0*0.0], [-X0/2,  -Y0,  Z0*0.0], [-X0,  -Y0/2,  Z0*0.0], [-X0/2,  -Y0/2,  Z0*0.0],
                   [X0,  -Y0,  Z0], [X0/2,  -Y0,  Z0], [X0,  -Y0/2,  Z0], [X0/2,  -Y0/2,  Z0],
                   [X0,  Y0,  Z0], [X0/2,  Y0,  Z0], [X0,  Y0/2,  Z0], [X0/2,  Y0/2,  Z0]])"""
x0_vec = np.array([x00, x01, x02, x03])
# x0_vec = np.array([x00])
# final state
xf = np.array([0, 0, 0])
# synthesis parameters

# initial population creation
# ----------------parse matrix basic solution parameters----------------------#

b_space = 5  # amount of columns to encode de basic solution
free_space = 10  # amount of columns to change the basic solution
colCount = b_space + free_space
rowCount = 6

# ----------------------------------------------------------------#
# -------initial population creation (small variation+basic_solution)-------- #
# population parameters
max_gens = 10
stop_score = 1.5
pop_size = 2**5
svRowCount = 10
# permitted_cols = [i for i in range(b_space + 1, colCount + 1)]
permitted_cols = [i for i in range(1, colCount + 1)]
permitted_rows = [i for i in range(1, rowCount + 1)]
sv_pop = np.zeros((pop_size, svRowCount, 4), dtype=int)
q_pop = np.zeros((pop_size, qCount))
q_min = 0
q_max = 10
elite_count = 3  # the amount of elite solutions that will be kept
group_count = (
    pop_size  # the amount of solutions that will be compared to the elite solutions
)
min_delta = 0.1
# ---------------------------------------------------------------------------------------#
# initialize the elite population with whatever
elite_svs = sv_pop[:elite_count]
elite_qs = q_pop[:elite_count]
elite_scores = np.full(elite_count, 2 + h * samples + 10)

# crossover parameters
cross_return_count = 2  # amount of spring that is returned in every crossover operation
cross_pop = pop_size
assert (
    cross_pop % cross_return_count
) == 0  # ensure that the crossover population can be generated in an integer
# amount of cycles
current_generation = 0
x0_index = len(x0_vec)
last_best_score = np.inf
gens_stuck = 0
max_gens_stuck = 5
restart_gens = 16
restart_count = 0
basic_solution = encode_u_simple(free_space, colCount, xCount, qCount)
"""basic_solution = np.array([encode_u1(free_space, colCount, xCount, qCount),
                           encode_u2(free_space, colCount, xCount, qCount)])"""
# ---------------itial population creation and score calculation--------------#
for i in range(pop_size):
    # population creation
    sv_pop[i] = encodeSVMat(
        permitted_rows, permitted_cols, xCount, uCount, qCount, svRowCount
    )
    q_pop[i] = encode_q(qCount, q_min, q_max)

# -------------------------score calculation--------------------------#
i_f, dis_to_xf, scores = group_score_calculation(
    x0_vec[:x0_index],
    samples,
    xCount,
    robot.model,
    h,
    Methods.euler,
    qCount,
    colCount,
    xf,
    u_min,
    u_max,
    uCount,
    min_delta,
    pop_size,
    sv_pop,
    q_pop,
    svRowCount,
    basic_solution,
)
scores_sorted_indexes = np.argsort(scores)
# -----------------------Genetic algorithm-------------------------------#
while True:
    restart_count += 1
    sorted_sv_group = sv_pop[scores_sorted_indexes[:group_count]]
    sorted_q_group = q_pop[scores_sorted_indexes[:group_count]]
    sorted_score_group = scores[scores_sorted_indexes[:group_count]]
    current_best_score = scores[scores_sorted_indexes[0]]
    if current_best_score == last_best_score:
        gens_stuck += 1
    else:
        gens_stuck = 0
    last_best_score = current_best_score

    elite_svs, elite_qs, elite_scores = stack_group_elite(
        sorted_sv_group,
        sorted_q_group,
        sorted_score_group,
        elite_svs,
        elite_qs,
        elite_scores,
        elite_count,
        group_count,
    )

    plot_x0_vec_trajectories(
        elite_svs[0],
        xCount,
        qCount,
        colCount,
        x0_vec[:x0_index],
        samples,
        h,
        robot.model,
        Methods.euler,
        elite_qs[0],
        xf,
        u_min,
        u_max,
        uCount,
        min_delta,
        svRowCount,
        basic_solution,
    )
    print(
        "gen:",
        current_generation,
        "elite score:",
        elite_scores,
        "gen_score: ",
        current_best_score,
        "gens_stuck:",
        gens_stuck,
    )
    if current_generation >= max_gens or elite_scores[0] < stop_score:
        break

    if restart_count >= restart_gens:
        basic_solution = applySVMat(basic_solution, elite_svs[0], svRowCount)
        # -------------------------initial population creation and score calculation--------------------------#
        for i in range(pop_size):
            # population creation
            sv_pop[i] = encodeSVMat(
                permitted_rows, permitted_cols, xCount, uCount, qCount, svRowCount
            )
            q_pop[i] = encode_q(qCount, q_min, q_max)
            # -------------------------score calculation--------------------------#

        i_f, dis_to_xf, scores = group_score_calculation(
            x0_vec[:x0_index],
            samples,
            xCount,
            robot.model,
            h,
            Methods.euler,
            qCount,
            colCount,
            xf,
            u_min,
            u_max,
            uCount,
            min_delta,
            pop_size,
            sv_pop,
            q_pop,
            svRowCount,
            basic_solution,
        )
        scores_sorted_indexes = np.argsort(scores)
        restart_count = 0
    else:
        # probabilities calculation
        probabilities = sum_based_prob(scores)
        # -------------------------crossover cycle--------------------------#
        sv_pop_new, q_pop_new, scores_new = crossover_cycle(
            sv_pop,
            q_pop,
            samples,
            probabilities,
            cross_return_count,
            cross_pop,
            svRowCount,
            qCount,
            x0_vec[:x0_index],
            xCount,
            robot.model,
            h,
            Methods.euler,
            colCount,
            xf,
            u_min,
            u_max,
            uCount,
            min_delta,
            svRowCount,
            basic_solution,
        )

        # passing to the new generation
        sv_pop = sv_pop_new.copy()
        q_pop = q_pop_new.copy()
        scores = scores_new.copy()
        scores_sorted_indexes = np.argsort(scores)
        current_generation += 1
# ---------------------------------------------------------------------------------------#
plot_x0_vec_trajectories(
    elite_svs[0],
    xCount,
    qCount,
    colCount,
    x0_vec[:x0_index],
    samples,
    h,
    robot.model,
    Methods.euler,
    elite_qs[0],
    xf,
    u_min,
    u_max,
    uCount,
    min_delta,
    svRowCount,
    basic_solution,
)
