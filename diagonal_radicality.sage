# ================================================================
# diagonal_radicality.sage
#
# Run from a shell:
#
#     sage diagonal_radicality.sage
#
# ================================================================

import os
import gc
import sys
import time
import traceback
import subprocess

from itertools import combinations
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ThreadPoolExecutor, as_completed

from sage.rings.polynomial.term_order import TermOrder

sys.setrecursionlimit(50000)
# ================================================================
# User parameters
# ================================================================

LAMBDA = (2,2,2,1)
M = 2

# Choices:
#     "paper"       block order of Definition 5.9
#     "degrevlex"   one global graded reverse lexicographic order
#     "deglex"      one global graded lexicographic order
#     "lex"         one global lexicographic order
#
# The collision ideal, degree-filtering, and defects use:
COLLISION_MONOMIAL_ORDER = "degrevlex"

# The ideals I_k and membership tests of defects use:
FILTRATION_MONOMIAL_ORDER = "degrevlex"

# Typical choices:
#     None
#     "singular:std"
#     "singular:slimgb"
#     "libsingular:std"
GB_ALGORITHM = "libsingular:std"

# Parallel computation of independent shape ideals.
PARALLEL_SHAPE_COMPUTATION = True
MAX_SHAPE_WORKERS = 3
SAGE_COMMAND = "sage"

REUSE_SAVED_SHAPE_IDEALS = True
COMPUTE_GB_FOR_SHAPE_IDEALS = True
COMPUTE_GB_FOR_FINAL_COLLISION_IDEAL = True

PRINT_MONOMIAL_PROGRESS = True
MONOMIAL_PROGRESS_INTERVAL = 100

SCRIPT_PATH = os.environ.get(
    "RADICALITY_SCRIPT_PATH",
    os.path.realpath(sys.argv[0])
)


# ================================================================
# Utilities
# ================================================================

def flush_output():
    try:
        sys.stdout.flush()
    except Exception:
        pass


def elapsed_string(start):
    s = time.perf_counter() - start
    d = int(s // 86400)
    s -= 86400 * d
    h = int(s // 3600)
    s -= 3600 * h
    m = int(s // 60)
    s -= 60 * m

    if d:
        return "%s days, %s hours, %s minutes, %.2f seconds" % (
            d, h, m, s
        )
    if h:
        return "%s hours, %s minutes, %.2f seconds" % (
            h, m, s
        )
    if m:
        return "%s minutes, %.2f seconds" % (m, s)
    return "%.2f seconds" % s


def order_tag(order):
    tags = {
        "paper": "paper_block_degrevlex",
        "degrevlex": "global_degrevlex",
        "deglex": "global_deglex",
        "lex": "global_lex"
    }

    if order not in tags:
        raise ValueError("Unknown monomial order: %s" % order)

    return tags[order]


def shape_to_string(shape):
    return "_".join(str(a) for a in shape)


def parse_partition_argument(text):
    return tuple(
        int(a) for a in text.split(",") if a != ""
    )


def ideal_statistics(I):
    G = list(I.gens())

    if len(G) == 0:
        return 0, 0

    return len(G), max(g.total_degree() for g in G)


# ================================================================
# Polynomial rings
# ================================================================

def diagonal_ring(m, n, monomial_order):
    names = []
    pos = {}
    counter = 0

    for i in range(n, 0, -1):
        for r in range(m):
            names.append("x_%s_%s" % (r + 1, i))
            pos[(r, i)] = counter
            counter += 1

    if monomial_order == "paper":
        order = TermOrder("degrevlex", m)

        for j in range(n - 1):
            order = order + TermOrder("degrevlex", m)

        R = PolynomialRing(QQ, names=names, order=order)

    elif monomial_order in ("degrevlex", "deglex", "lex"):
        R = PolynomialRing(
            QQ,
            names=names,
            order=monomial_order
        )

    else:
        raise ValueError(
            "Unknown monomial order: %s"
            % monomial_order
        )

    gens = R.gens()

    x = [[None for i in range(n + 1)] for r in range(m)]

    for r in range(m):
        for i in range(1, n + 1):
            x[r][i] = gens[pos[(r, i)]]

    return R, x, pos


def monomial_from_exponents(R, exponents):
    ans = R.one()

    for variable, exponent in zip(R.gens(), exponents):
        if exponent:
            ans *= variable**exponent

    return ans


def transfer_polynomial(f, target_ring):
    """
    Transfer a polynomial to a ring with the same ordered variable list
    but possibly a different monomial order.
    """
    ans = target_ring.zero()

    for exponent_vector, coefficient in f.dict().items():
        ans += QQ(coefficient) * monomial_from_exponents(
            target_ring,
            exponent_vector
        )

    return ans


def transfer_polynomial_list(polynomials, target_ring):
    return [
        transfer_polynomial(f, target_ring)
        for f in polynomials
    ]


def normal_form(f, G):
    if len(G) == 0:
        return f

    return f.reduce(G)


# ================================================================
# Symmetric group action
# ================================================================

def act_on_monomial(mon, sigma, R, m, n, pos):
    old = mon.exponents()[0]
    new = [0 for j in range(R.ngens())]

    for r in range(m):
        for i in range(1, n + 1):
            new[pos[(r, sigma(i))]] = old[pos[(r, i)]]

    return monomial_from_exponents(R, new)


def act_on_polynomial(f, sigma, R, m, n, pos):
    ans = R.zero()

    for exponents, coefficient in f.dict().items():
        mon = monomial_from_exponents(R, exponents)

        ans += coefficient * act_on_monomial(
            mon, sigma, R, m, n, pos
        )

    return ans


def cycle_type_of_permutation(sigma):
    return Partition(
        sorted(sigma.cycle_type(), reverse=True)
    )


# ================================================================
# Minimal shapes
# ================================================================

def dominates_partition(lam, mu):
    L = max(len(lam), len(mu))

    lam = list(lam) + [0] * (L - len(lam))
    mu = list(mu) + [0] * (L - len(mu))

    a = 0
    b = 0

    for i in range(L):
        a += lam[i]
        b += mu[i]

        if a < b:
            return False

    return True


def refines_partition(fine, coarse):
    if sum(fine) != sum(coarse):
        return False

    fine = sorted(fine, reverse=True)
    sums = [0 for j in coarse]

    def recurse(i):
        if i == len(fine):
            return all(
                sums[j] == coarse[j]
                for j in range(len(coarse))
            )

        tried = set()
        part = fine[i]

        for j in range(len(coarse)):
            state = (coarse[j], sums[j])

            if state in tried:
                continue

            tried.add(state)

            if sums[j] + part <= coarse[j]:
                sums[j] += part

                if recurse(i + 1):
                    return True

                sums[j] -= part

        return False

    return recurse(0)


def minimal_shapes_for_lambda(lam):
    n = sum(lam)

    candidates = [
        tuple(mu)
        for mu in Partitions(n)
        if not dominates_partition(tuple(lam), tuple(mu))
    ]

    answer = []

    for mu in candidates:
        if not any(
            nu != mu and refines_partition(nu, mu)
            for nu in candidates
        ):
            answer.append(mu)

    return sorted(answer, reverse=True)


# ================================================================
# Characters and projections
# ================================================================

@cached_function
def z_mu(mu):
    mu = Partition(mu)
    multiplicities = mu.to_exp()
    ans = ZZ.one()

    for i, multiplicity in enumerate(
            multiplicities,
            start=1):

        if multiplicity:
            ans *= i**multiplicity * factorial(multiplicity)

    return ans


@cached_function
def character_value(lam, mu):
    Sym = SymmetricFunctions(QQ)
    s = Sym.schur()
    p = Sym.powersum()

    lam = Partition(lam)
    mu = Partition(mu)

    return ZZ(z_mu(mu) * p(s[lam])[mu])


def lambda_dimension(lam):
    return character_value(
        Partition(lam),
        Partition([1 for j in range(sum(lam))])
    )


def unnormalized_projection(
        f, lam, R, m, n, pos,
        permutations, character_values):

    ans = R.zero()

    for sigma in permutations:
        mu = cycle_type_of_permutation(sigma)
        chi = character_values[mu]

        if chi:
            ans += chi * act_on_polynomial(
                f, sigma, R, m, n, pos
            )

    return ans


def normalized_projection(
        f, lam, R, m, n, pos,
        permutations, character_values,
        dim_lambda):

    return (
        QQ(dim_lambda) / QQ(factorial(n))
    ) * unnormalized_projection(
        f, lam, R, m, n, pos,
        permutations, character_values
    )


# ================================================================
# Gröbner bases
# ================================================================

def selected_groebner_basis(I, description):
    print("")
    print("START Groebner basis:", description)
    print("Algorithm:", GB_ALGORITHM)
    flush_output()

    start = time.perf_counter()

    if GB_ALGORITHM is None:
        G = list(I.groebner_basis())
    else:
        G = list(
            I.groebner_basis(
                algorithm=GB_ALGORITHM
            )
        )

    print("END Groebner basis:", description)
    print("Number of elements:", len(G))
    print("Elapsed time:", elapsed_string(start))
    flush_output()

    return G


# ================================================================
# Collision primes
# ================================================================

def shape_of_set_partition(Pi):
    return tuple(
        sorted(
            [len(block) for block in Pi],
            reverse=True
        )
    )


def set_partitions_of_shape(n, shape):
    return [
        Pi for Pi in SetPartitions(n)
        if shape_of_set_partition(Pi) == tuple(shape)
    ]


def linear_prime_of_set_partition(R, x, Pi, m):
    generators = []

    for block in Pi:
        block = list(block)

        for i, j in combinations(block, 2):
            for r in range(m):
                generators.append(
                    x[r][i] - x[r][j]
                )

    return R.ideal(generators)


# ================================================================
# Intersections
# ================================================================

def intersect_ideals_balanced(ideals, label):
    current = list(ideals)
    level = 0
    start = time.perf_counter()

    while len(current) > 1:
        level += 1

        print("")
        print(
            "%s: balanced intersection level %s."
            % (label, level)
        )
        print("Number of ideals:", len(current))
        flush_output()

        next_level = []

        for j in range(0, len(current) - 1, 2):
            pair_start = time.perf_counter()

            print(
                "%s: START pair %s."
                % (label, j // 2 + 1)
            )
            flush_output()

            K = current[j].intersection(current[j + 1])
            number, degree = ideal_statistics(K)

            print(
                "%s: END pair %s."
                % (label, j // 2 + 1)
            )
            print("Stored generators:", number)
            print("Maximum generator degree:", degree)
            print("Pair elapsed:", elapsed_string(pair_start))
            flush_output()

            next_level.append(K)

        if len(current) % 2:
            next_level.append(current[-1])

        del current
        gc.collect()
        current = next_level

    print("")
    print(label + ": complete.")
    print("Total intersection time:", elapsed_string(start))
    flush_output()

    return current[0]


# ================================================================
# State paths
# ================================================================

def state_directory_name(lam, m):
    lam_string = "_".join(str(a) for a in lam)

    return (
        "saved_states_lambda_%s_m_%s"
        "_collision_%s_filtration_%s"
        % (
            lam_string,
            m,
            order_tag(COLLISION_MONOMIAL_ORDER),
            order_tag(FILTRATION_MONOMIAL_ORDER)
        )
    )


def shape_state_filename(shape, directory):
    return os.path.join(
        directory,
        "collision_shape_%s_order_%s.sobj"
        % (
            shape_to_string(shape),
            order_tag(COLLISION_MONOMIAL_ORDER)
        )
    )


def collision_state_filename(lam, m, directory):
    return os.path.join(
        directory,
        "collision_final_order_%s.sobj"
        % order_tag(COLLISION_MONOMIAL_ORDER)
    )


def degree_state_filename(k, directory):
    return os.path.join(
        directory,
        "filtration_order_%s_degree_%s.sobj"
        % (
            order_tag(FILTRATION_MONOMIAL_ORDER),
            k
        )
    )


# ================================================================
# Shape ideal computation
# ================================================================

def compute_shape_collision_ideal(
        R, x, m, n, shape, directory):

    filename = shape_state_filename(shape, directory)

    if (
        REUSE_SAVED_SHAPE_IDEALS
        and os.path.exists(filename)
    ):
        state = load(filename)

        if state["collision_order"] != COLLISION_MONOMIAL_ORDER:
            raise RuntimeError(
                "Wrong monomial order in saved shape state."
            )

        print("Reusing saved shape ideal:", shape)
        print("Stored generators:", state["generator_count"])
        print(
            "Stored maximum degree:",
            state["maximum_generator_degree"]
        )
        print("Stored elapsed time:", state["elapsed_time"])
        flush_output()

        J_shape = state["J_shape"]

        del state
        gc.collect()

        return J_shape

    print("")
    print("======================================================")
    print("Computing shape ideal:", shape)
    print("Collision order:", COLLISION_MONOMIAL_ORDER)
    print("======================================================")
    flush_output()

    start = time.perf_counter()
    partitions = set_partitions_of_shape(n, shape)

    print("Number of collision primes:", len(partitions))
    flush_output()

    primes = []

    for j, Pi in enumerate(partitions, start=1):
        if j % 25 == 0:
            print("Constructed %s collision primes." % j)
            flush_output()

        primes.append(
            linear_prime_of_set_partition(R, x, Pi, m)
        )

    del partitions
    gc.collect()

    J_shape = intersect_ideals_balanced(
        primes,
        "Shape %s" % (shape,)
    )

    del primes
    gc.collect()

    if COMPUTE_GB_FOR_SHAPE_IDEALS:
        G = selected_groebner_basis(
            J_shape,
            "shape ideal %s" % (shape,)
        )

        J_shape = R.ideal(G)

        del G
        gc.collect()

    number, degree = ideal_statistics(J_shape)
    total_time = elapsed_string(start)

    state = {
        "J_shape": J_shape,
        "shape": tuple(shape),
        "m": m,
        "n": n,
        "collision_order": COLLISION_MONOMIAL_ORDER,
        "generator_count": number,
        "maximum_generator_degree": degree,
        "elapsed_time": total_time
    }

    print("Saving shape state:", filename)
    flush_output()

    save(state, filename)

    print("Shape state saved.")
    print("Total shape time:", total_time)
    flush_output()

    del state
    gc.collect()

    return J_shape


# ================================================================
# Parallel workers
# ================================================================

def shape_worker_main(lam, m, shape, directory):
    n = sum(lam)
    lam_string = "_".join(str(a) for a in lam)

    log_filename = os.path.join(
        directory,
        "worker_lambda_%s_m_%s_shape_%s_collision_%s.output"
        % (
            lam_string,
            m,
            shape_to_string(shape),
            order_tag(COLLISION_MONOMIAL_ORDER)
        )
    )

    with open(log_filename, "w", buffering=1) as worker_log:
        with redirect_stdout(worker_log):
            with redirect_stderr(worker_log):

                print("Shape worker")
                print("lambda =", tuple(lam))
                print("m =", m)
                print("shape =", tuple(shape))
                print("collision order =",
                      COLLISION_MONOMIAL_ORDER)
                flush_output()

                R, x, pos = diagonal_ring(
                    m,
                    n,
                    COLLISION_MONOMIAL_ORDER
                )

                compute_shape_collision_ideal(
                    R, x, m, n, shape, directory
                )

                print("Shape worker completed.")
                flush_output()


def launch_shape_worker(lam, m, shape, directory):
    env = os.environ.copy()

    env["RADICALITY_SHAPE_WORKER"] = "1"
    env["RADICALITY_WORKER_LAMBDA"] = ",".join(
        str(a) for a in lam
    )
    env["RADICALITY_WORKER_M"] = str(m)
    env["RADICALITY_WORKER_SHAPE"] = ",".join(
        str(a) for a in shape
    )
    env["RADICALITY_WORKER_DIRECTORY"] = os.path.abspath(
        directory
    )
    env["RADICALITY_SCRIPT_PATH"] = SCRIPT_PATH

    launcher_log = os.path.join(
        directory,
        "worker_launcher_shape_%s_collision_%s.log"
        % (
            shape_to_string(shape),
            order_tag(COLLISION_MONOMIAL_ORDER)
        )
    )

    with open(launcher_log, "w", buffering=1) as F:
        result = subprocess.run(
            [
                SAGE_COMMAND,
                "-c",
                "load(%r)" % SCRIPT_PATH
            ],
            env=env,
            stdout=F,
            stderr=subprocess.STDOUT
        )

    return tuple(shape), result.returncode, launcher_log


# ================================================================
# Collision ideal shape by shape
# ================================================================

def collision_ideal_by_shapes(
        R, x, m, n,
        minimal_shapes, directory):

    shape_data = sorted([
        (
            len(set_partitions_of_shape(n, shape)),
            tuple(shape)
        )
        for shape in minimal_shapes
    ])

    print("")
    print("======================================================")
    print("Collision ideal computation by shapes")
    print("======================================================")
    print("Collision order:", COLLISION_MONOMIAL_ORDER)

    for count, shape in shape_data:
        print(
            "shape %s: %s collision primes"
            % (shape, count)
        )

    flush_output()

    if PARALLEL_SHAPE_COMPUTATION and len(shape_data) > 1:
        missing = []

        for count, shape in shape_data:
            if not (
                REUSE_SAVED_SHAPE_IDEALS
                and os.path.exists(
                    shape_state_filename(shape, directory)
                )
            ):
                missing.append(shape)

        if len(missing):
            workers = min(MAX_SHAPE_WORKERS, len(missing))

            print(
                "Launching %s simultaneous shape workers."
                % workers
            )
            flush_output()

            with ThreadPoolExecutor(
                    max_workers=workers) as executor:

                futures = {
                    executor.submit(
                        launch_shape_worker,
                        tuple(LAMBDA),
                        m,
                        shape,
                        directory
                    ): shape
                    for shape in missing
                }

                for future in as_completed(futures):
                    shape, code, logfile = future.result()

                    print(
                        "Worker finished for shape %s "
                        "with return code %s."
                        % (shape, code)
                    )
                    print("Worker launcher log:", logfile)
                    flush_output()

                    if code != 0:
                        raise RuntimeError(
                            "Shape worker failed for %s. "
                            "Inspect %s"
                            % (shape, logfile)
                        )

        shape_ideals = []

        for count, shape in shape_data:
            state = load(
                shape_state_filename(shape, directory)
            )

            if state["collision_order"] != COLLISION_MONOMIAL_ORDER:
                raise RuntimeError(
                    "Wrong collision order in saved shape state."
                )

            shape_ideals.append(state["J_shape"])

            print(
                "Loaded shape %s with %s generators."
                % (
                    shape,
                    state["generator_count"]
                )
            )
            flush_output()

            del state
            gc.collect()

    else:
        shape_ideals = [
            compute_shape_collision_ideal(
                R, x, m, n, shape, directory
            )
            for count, shape in shape_data
        ]

    print("")
    print("Final intersection of shape ideals.")
    flush_output()

    J = intersect_ideals_balanced(
        shape_ideals,
        "Final collision ideal"
    )

    del shape_ideals
    gc.collect()

    if COMPUTE_GB_FOR_FINAL_COLLISION_IDEAL:
        G = selected_groebner_basis(
            J,
            "final collision ideal"
        )

        J = R.ideal(G)

        del G
        gc.collect()

    return J, shape_data


# ================================================================
# Monomial orbit representatives
# ================================================================

def column_profiles(m, d):
    ans = []

    for e in range(d + 1):
        for alpha in IntegerVectors(e, m):
            ans.append(tuple(alpha))

    return ans


def monomial_from_profiles(R, profiles, m, n, pos):
    exponents = [0 for j in range(R.ngens())]

    for i in range(1, n + 1):
        for r in range(m):
            exponents[pos[(r, i)]] = profiles[i - 1][r]

    return monomial_from_exponents(R, exponents)


def monomial_orbit_representatives(R, m, n, d, pos):
    profiles = column_profiles(m, d)
    degrees = [sum(profile) for profile in profiles]
    chosen = []

    def recurse(start, columns_left, degree_left):
        if columns_left == 0:
            if degree_left == 0:
                yield list(chosen)
            return

        for j in range(start, len(profiles)):
            if degrees[j] > degree_left:
                break

            chosen.append(profiles[j])

            yield from recurse(
                j,
                columns_left - 1,
                degree_left - degrees[j]
            )

            chosen.pop()

    for profile_list in recurse(0, n, d):
        yield monomial_from_profiles(
            R, profile_list, m, n, pos
        )


def polynomial_key(f):
    return tuple(sorted(
        (
            tuple(exponents),
            QQ(coefficient)
        )
        for exponents, coefficient in f.dict().items()
        if coefficient != 0
    ))


def polynomial_orbit(f, R, m, n, pos, permutations):
    orbit = []
    seen = set()

    for sigma in permutations:
        g = act_on_polynomial(
            f, sigma, R, m, n, pos
        )

        key = polynomial_key(g)

        if key not in seen:
            seen.add(key)
            orbit.append(g)

    return orbit


# ================================================================
# Degree-filtered collision generators
# ================================================================

def degree_filtered_generator_reduction(R, G_all):
    remaining = list(G_all)
    selected = []
    K = R.ideal([])

    D = max(g.total_degree() for g in G_all)

    print("")
    print("======================================================")
    print("Degree-filtered generator reduction")
    print("======================================================")
    print("Initial generators:", len(remaining))
    print("Maximum degree:", D)
    flush_output()

    for d in range(1, D + 1):
        if len(remaining) == 0:
            break

        GK_old = selected_groebner_basis(
            K,
            "degree-filtered ideal before degree %s" % d
        )

        candidates = [
            g for g in remaining
            if (
                g.total_degree() == d
                and normal_form(g, GK_old) != 0
            )
        ]

        if len(candidates):
            K = R.ideal(GK_old + candidates)

        GK = selected_groebner_basis(
            K,
            "degree-filtered ideal after degree %s" % d
        )

        new_remaining = [
            g for g in remaining
            if normal_form(g, GK) != 0
        ]

        print("Degree", d)
        print("New generators:", len(candidates))
        print(
            "Generators removed:",
            len(remaining) - len(new_remaining)
        )
        print("Generators remaining:", len(new_remaining))
        flush_output()

        selected.extend(candidates)
        remaining = new_remaining

        del GK_old
        del GK
        del candidates
        gc.collect()

    if len(remaining):
        raise RuntimeError(
            "Degree-filtered reduction failed."
        )

    return selected, K


# ================================================================
# State saving
# ================================================================

def save_collision_state(
        lam, m, n, minimal_shapes,
        J, G, shape_data, directory):

    filename = collision_state_filename(lam, m, directory)

    state = {
        "J": J,
        "G": G,
        "lambda": tuple(lam),
        "m": m,
        "n": n,
        "minimal_shapes": minimal_shapes,
        "shape_data": shape_data,
        "collision_order": COLLISION_MONOMIAL_ORDER,
        "filtration_order": FILTRATION_MONOMIAL_ORDER
    }

    print("Saving collision state:", filename)
    flush_output()
    save(state, filename)
    del state
    gc.collect()


def save_degree_state(
        lam, m, n, k, I, GI,
        remaining_defects, directory):

    filename = degree_state_filename(k, directory)

    state = {
        "lambda": tuple(lam),
        "m": m,
        "n": n,
        "degree": k,
        "I": I,
        "GI": GI,
        "remaining_defects": remaining_defects,
        "collision_order": COLLISION_MONOMIAL_ORDER,
        "filtration_order": FILTRATION_MONOMIAL_ORDER
    }

    print("Saving degree state:", filename)
    flush_output()
    save(state, filename)
    del state
    gc.collect()


# ================================================================
# Main radicality computation
# ================================================================

def radicality_test_by_degree(lam, m):
    total_start = time.perf_counter()
    lam = Partition(lam)
    n = sum(lam)

    directory = state_directory_name(lam, m)
    os.makedirs(directory, exist_ok=True)

    print("")
    print("======================================================")
    print("Radicality computation")
    print("lambda =", tuple(lam))
    print("m =", m)
    print("n =", n)
    print("collision order =", COLLISION_MONOMIAL_ORDER)
    print("filtration order =", FILTRATION_MONOMIAL_ORDER)
    print("======================================================")
    flush_output()

    minimal_shapes = minimal_shapes_for_lambda(lam)

    print("Minimal shapes:", minimal_shapes)
    print("State directory:", directory)
    flush_output()

    # ------------------------------------------------------------
    # Collision ideal, filtered generators, and defects:
    # all in collision-order ring.
    # ------------------------------------------------------------
    R_collision, x_collision, pos_collision = diagonal_ring(
        m, n, COLLISION_MONOMIAL_ORDER
    )

    J_collision, shape_data = collision_ideal_by_shapes(
        R_collision,
        x_collision,
        m,
        n,
        minimal_shapes,
        directory
    )

    if COMPUTE_GB_FOR_FINAL_COLLISION_IDEAL:
        G_collision = list(J_collision.gens())
    else:
        G_collision = selected_groebner_basis(
            J_collision,
            "collision ideal"
        )
        J_collision = R_collision.ideal(G_collision)

    if len(G_collision) == 0:
        print("J=(0). Therefore I_lambda is radical.")
        return

    D_full = max(g.total_degree() for g in G_collision)

    print("Collision generators:", len(G_collision))
    print("D_full =", D_full)
    flush_output()

    G_filtered, J_filtered = degree_filtered_generator_reduction(
        R_collision,
        G_collision
    )

    D_filtered = max(g.total_degree() for g in G_filtered)

    print("D_filtered =", D_filtered)
    flush_output()

    del J_filtered
    gc.collect()

    Sn = SymmetricGroup(n)
    permutations = list(Sn)

    conjugacy_classes = sorted(
        set(
            cycle_type_of_permutation(sigma)
            for sigma in permutations
        ),
        reverse=True
    )

    character_values = {
        mu: character_value(lam, mu)
        for mu in conjugacy_classes
    }

    dim_lambda = lambda_dimension(lam)

    print("Dimension of S^lambda:", dim_lambda)
    flush_output()

    fixed_collision = []
    defects_collision = []

    for j, f in enumerate(G_filtered, start=1):
        print(
            "Collision-ring projection %s of %s."
            % (j, len(G_filtered))
        )
        flush_output()

        Pf = normalized_projection(
            f,
            lam,
            R_collision,
            m,
            n,
            pos_collision,
            permutations,
            character_values,
            dim_lambda
        )

        defect = f - Pf

        if defect == 0:
            fixed_collision.append(f)
        else:
            defects_collision.append({
                "source_generator": f,
                "defect": defect,
                "degree": f.total_degree()
            })

        del Pf
        gc.collect()

    print("Projection-fixed generators:", len(fixed_collision))
    print("Nonzero defects:", len(defects_collision))
    flush_output()

    save_collision_state(
        lam,
        m,
        n,
        minimal_shapes,
        J_collision,
        G_collision,
        shape_data,
        directory
    )

    if len(defects_collision) == 0:
        print("")
        print("======================================================")
        print("RESULT: I_lambda IS RADICAL")
        print("======================================================")
        return

    # ------------------------------------------------------------
    # Transfer only defects to filtration-order ring.
    # ------------------------------------------------------------
    R, x, pos = diagonal_ring(
        m, n, FILTRATION_MONOMIAL_ORDER
    )

    remaining_defects = []

    for item in defects_collision:
        remaining_defects.append({
            "source_generator": transfer_polynomial(
                item["source_generator"], R
            ),
            "defect": transfer_polynomial(
                item["defect"], R
            ),
            "degree": item["degree"]
        })

    del fixed_collision
    del defects_collision
    del G_filtered
    del G_collision
    del J_collision
    del R_collision
    del x_collision
    del pos_collision
    gc.collect()

    D_defects = max(
        item["degree"]
        for item in remaining_defects
    )

    print("D_defects =", D_defects)
    flush_output()

    for item in remaining_defects:
        if item["degree"] == 1:
            print("")
            print("======================================================")
            print("RESULT: I_lambda IS NOT RADICAL")
            print("======================================================")
            print("Witness:")
            print(item["defect"])
            return

    # ------------------------------------------------------------
    # Construct filtration I_k in filtration-order ring.
    # ------------------------------------------------------------
    I = R.ideal([])

    for k in range(1, D_defects):
        print("")
        print("======================================================")
        print("Degree k =", k)
        print("======================================================")
        flush_output()

        GI_old = selected_groebner_basis(
            I,
            "I_%s" % (k - 1)
        )

        representatives = list(
            monomial_orbit_representatives(
                R, m, n, k, pos
            )
        )

        print("Monomial orbits:", len(representatives))
        flush_output()

        new_generators = []
        nonzero = 0
        already_present = 0

        for j, mon in enumerate(representatives, start=1):
            if (
                PRINT_MONOMIAL_PROGRESS
                and j % MONOMIAL_PROGRESS_INTERVAL == 0
            ):
                print(
                    "Processed %s of %s representatives."
                    % (j, len(representatives))
                )
                flush_output()

            projection = unnormalized_projection(
                mon,
                lam,
                R,
                m,
                n,
                pos,
                permutations,
                character_values
            )

            if projection == 0:
                continue

            nonzero += 1

            if normal_form(projection, GI_old) == 0:
                already_present += 1
                continue

            new_generators.extend(
                polynomial_orbit(
                    projection,
                    R,
                    m,
                    n,
                    pos,
                    permutations
                )
            )

        print("Nonzero projections:", nonzero)
        print("Already in I_(k-1):", already_present)
        print("New orbit generators:", len(new_generators))
        flush_output()

        I = R.ideal(GI_old + new_generators)

        GI = selected_groebner_basis(
            I,
            "I_%s" % k
        )

        witness = None
        new_remaining = []
        removed = 0

        print(
            "Testing all remaining defects in I_%s."
            % k
        )
        flush_output()

        for item in remaining_defects:
            remainder = normal_form(item["defect"], GI)

            if remainder == 0:
                removed += 1
                continue

            if item["degree"] == k + 1:
                witness = item["defect"]
                break

            new_remaining.append(item)

        print("Defects removed:", removed)
        flush_output()

        if witness is None:
            remaining_defects = new_remaining

        save_degree_state(
            lam,
            m,
            n,
            k,
            I,
            GI,
            remaining_defects,
            directory
        )

        del representatives
        del new_generators
        del GI_old
        del new_remaining
        gc.collect()

        if witness is not None:
            print("")
            print("======================================================")
            print("RESULT: I_lambda IS NOT RADICAL")
            print("======================================================")
            print("Witness:")
            print(witness)
            return

        if len(remaining_defects) == 0:
            print("")
            print("======================================================")
            print("RESULT: I_lambda IS RADICAL")
            print("======================================================")
            print(
                "Every nonzero defect belongs to I_%s."
                % k
            )
            print(
                "Total elapsed time:",
                elapsed_string(total_start)
            )
            return

    if len(remaining_defects):
        raise RuntimeError("Some defects were unresolved.")

    print("")
    print("======================================================")
    print("RESULT: I_lambda IS RADICAL")
    print("======================================================")
    print(
        "Total elapsed time:",
        elapsed_string(total_start)
    )


# ================================================================
# Main process
# ================================================================

def run_computation():
    lam_string = "_".join(str(a) for a in LAMBDA)

    output_filename = (
        "lambda_%s_m_%s_collision_%s_filtration_%s.output"
        % (
            lam_string,
            M,
            order_tag(COLLISION_MONOMIAL_ORDER),
            order_tag(FILTRATION_MONOMIAL_ORDER)
        )
    )

    with open(output_filename, "w", buffering=1) as output:
        with redirect_stdout(output):
            with redirect_stderr(output):

                try:
                    print("======================================================")
                    print("Parallel radicality computation")
                    print("======================================================")
                    print("Input lambda =", LAMBDA)
                    print("Input m =", M)
                    print("n =", sum(LAMBDA))
                    print(
                        "Collision monomial order =",
                        COLLISION_MONOMIAL_ORDER
                    )
                    print(
                        "Filtration monomial order =",
                        FILTRATION_MONOMIAL_ORDER
                    )
                    print("GB algorithm =", GB_ALGORITHM)
                    print(
                        "Maximum shape workers =",
                        MAX_SHAPE_WORKERS
                    )
                    flush_output()

                    radicality_test_by_degree(LAMBDA, M)

                    print("")
                    print("Computation completed.")
                    flush_output()

                except Exception:
                    print("")
                    print("======================================================")
                    print("EXCEPTION / TRACEBACK")
                    print("======================================================")
                    traceback.print_exc(file=output)
                    output.flush()
                    raise

    print("Computation finished.")
    print("Output written to:", output_filename)


# ================================================================
# Entry point
# ================================================================

if os.environ.get("RADICALITY_SHAPE_WORKER") == "1":

    worker_lambda = parse_partition_argument(
        os.environ["RADICALITY_WORKER_LAMBDA"]
    )

    worker_m = int(
        os.environ["RADICALITY_WORKER_M"]
    )

    worker_shape = parse_partition_argument(
        os.environ["RADICALITY_WORKER_SHAPE"]
    )

    worker_directory = os.environ[
        "RADICALITY_WORKER_DIRECTORY"
    ]

    os.makedirs(worker_directory, exist_ok=True)

    shape_worker_main(
        worker_lambda,
        worker_m,
        worker_shape,
        worker_directory
    )

else:
    run_computation()