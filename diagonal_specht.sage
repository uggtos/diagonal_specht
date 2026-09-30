# -*- python-indent-offset: 4; indent-tabs-mode: nil -*-
# ================================================================
# diagonal_specht.sage
#
# Exact SageMath computations over QQ for:
#
# (1) Lemma 5.14:
#       I(V_(1,1,1)) = <B^(l)> for 1 <= l <= 3;
#       every S-polynomial of two elements of B^(l) reduces to zero
#       modulo B^(l), for 1 <= l <= 6.
#
# (2) Radicality:
#       I_(2,2,1) in R_(2,5) is radical;
#       I_(3,2,1) in R_(2,6) is radical.
#
# The polynomial-ring order is the order of Definition 5.9.
#
# All references are according to the original article
#
# ================================================================


from itertools import combinations
from sage.rings.polynomial.term_order import TermOrder


# ================================================================
# Optional Singular S-polynomial routine
# ================================================================

# Some Sage versions expose Singular's spoly procedure directly,
# while others do not.  We attempt to use it first.  If unavailable,
# the script uses the standard explicit definition of S-polynomials.

USE_SINGULAR_SPOLY = False
spoly_singular = None
singular = None

#try:
#    from sage.interfaces.singular import singular
#
#    # general.lib contains various standard utility procedures.
#    singular.lib("general.lib")
#
#    # On many Singular installations, this gives the spoly routine.
#    spoly_singular = singular.function("spoly")
#    USE_SINGULAR_SPOLY = True
#
#except Exception:
#    USE_SINGULAR_SPOLY = False
#    spoly_singular = None


# ================================================================
# Polynomial rings and the monomial order
# ================================================================

def diagonal_ring(m, n):
    r"""
    Return (R,x,pos), where

        R = QQ[x_{r,i} : 1 <= r <= m, 1 <= i <= n].

    The variables are listed in the order

        x_{1,n}, ..., x_{m,n},
        x_{1,n-1}, ..., x_{m,n-1},
        ...
        x_{1,1}, ..., x_{m,1}.

    The monomial order is a block order with degrevlex blocks:

        [column n] > [column n-1] > ... > [column 1].

    Inside each block,

        x_{1,i} > x_{2,i} > ... > x_{m,i}.

    This is the order of Definition 5.9.
    """
    names = []
    pos = {}
    counter = 0

    # Column n is compared first, hence it is listed first.
    for i in range(n, 0, -1):
        for r in range(m):
            names.append("x_%s_%s" % (r + 1, i))
            pos[(r, i)] = counter
            counter += 1

    order = TermOrder("degrevlex", m)

    for j in range(n - 1):
        order = order + TermOrder("degrevlex", m)

    R = PolynomialRing(QQ, names=names, order=order)
    variables = R.gens()

    x = [[None for i in range(n + 1)] for r in range(m)]

    for r in range(m):
        for i in range(1, n + 1):
            x[r][i] = variables[pos[(r, i)]]

    return R, x, pos


def monomial_from_exponents(R, exponents):
    """
    Construct a monomial in R from an exponent vector.
    """
    ans = R.one()

    for variable, exponent in zip(R.gens(), exponents):
        if exponent != 0:
            ans *= variable**exponent

    return ans


def monomials_of_degree_at_most(R, D):
    """
    Return all monomials in R of total degree at most D.
    """
    ans = []
    N = R.ngens()

    for d in range(D + 1):
        for exponent_vector in IntegerVectors(d, N):
            ans.append(monomial_from_exponents(R, exponent_vector))

    return ans


def normal_form(f, generators):
    """
    Reduce f by the ordered list of generators.
    """
    return f.reduce(generators)


def ideals_are_equal(I, J):
    """
    Test equality of ideals by reducing Groebner bases in both directions.
    """
    GI = list(I.groebner_basis())
    GJ = list(J.groebner_basis())

    return (
        all(normal_form(f, GJ) == 0 for f in GI)
        and
        all(normal_form(g, GI) == 0 for g in GJ)
    )


# ================================================================
# S-polynomials
# ================================================================

def monomial_lcm(a, b, R):
    """
    Least common multiple of two monomials.
    """
    exponent_a = a.exponents()[0]
    exponent_b = b.exponents()[0]

    exponent_lcm = [
        max(exponent_a[i], exponent_b[i])
        for i in range(len(exponent_a))
    ]

    return monomial_from_exponents(R, exponent_lcm)


def monomial_quotient(a, b, R):
    """
    Return a/b, assuming that the monomial b divides the monomial a.
    """
    exponent_a = a.exponents()[0]
    exponent_b = b.exponents()[0]

    assert all(
        exponent_a[i] >= exponent_b[i]
        for i in range(len(exponent_a))
    )

    exponent_quotient = [
        exponent_a[i] - exponent_b[i]
        for i in range(len(exponent_a))
    ]

    return monomial_from_exponents(R, exponent_quotient)


def explicit_s_polynomial(f, g, R):
    r"""
    Compute the S-polynomial directly:

       lcm(LM(f),LM(g))/LT(f) * f
       -
       lcm(LM(f),LM(g))/LT(g) * g.
    """
    lm_f = f.leading_monomial()
    lm_g = g.leading_monomial()

    lc_f = f.leading_coefficient()
    lc_g = g.leading_coefficient()

    lcm_fg = monomial_lcm(lm_f, lm_g, R)

    multiplier_f = monomial_quotient(lcm_fg, lm_f, R) / lc_f
    multiplier_g = monomial_quotient(lcm_fg, lm_g, R) / lc_g

    return multiplier_f * f - multiplier_g * g


def s_polynomial(f, g, R):
    r"""
    Compute the S-polynomial of f and g.

    We use Singular's built-in spoly routine whenever it is accessible.
    If not, we use the standard explicit formula.
    """
    global USE_SINGULAR_SPOLY
    global spoly_singular
    global singular

    if USE_SINGULAR_SPOLY:
        try:
            f_singular = singular(f)
            g_singular = singular(g)

            return R(spoly_singular(f_singular, g_singular))

        except Exception:
            # If Singular's interface fails on this Sage version,
            # permanently switch to the explicit implementation.
            USE_SINGULAR_SPOLY = False

    return explicit_s_polynomial(f, g, R)


# ================================================================
# Symmetric-group action
# ================================================================

def act_on_monomial(mon, permutation, R, m, n, pos):
    r"""
    Apply permutation to a monomial, with convention

        permutation . x_{r,i} = x_{r,permutation(i)}.
    """
    old_exponents = mon.exponents()[0]
    new_exponents = [0 for j in range(R.ngens())]

    for r in range(m):
        for i in range(1, n + 1):
            old_position = pos[(r, i)]
            new_position = pos[(r, permutation(i))]
            new_exponents[new_position] = old_exponents[old_position]

    return monomial_from_exponents(R, new_exponents)


def cycle_type_of_permutation(permutation):
    """
    Return the cycle type as a Sage partition.
    """
    return Partition(
        sorted(permutation.cycle_type(), reverse=True)
    )


# ================================================================
# Irreducible characters through Sage symmetric functions
# ================================================================

@cached_function
def z_mu(mu):
    r"""
    Compute

        z_mu = product_i i^(m_i) m_i!,

    where m_i is the number of parts of mu equal to i.
    """
    mu = Partition(mu)
    multiplicities = mu.to_exp()

    ans = ZZ.one()

    for i, multiplicity in enumerate(multiplicities, start=1):
        if multiplicity != 0:
            ans *= i**multiplicity * factorial(multiplicity)

    return ans


@cached_function
def character_value(lam, mu):
    r"""
    Return chi^lam(mu), computed using Sage's built-in
    symmetric-function machinery.

    The Frobenius characteristic formula says

        s_lam = sum_mu chi^lam(mu)/z_mu * p_mu.

    Therefore chi^lam(mu) is z_mu times the coefficient of p_mu
    in s_lam expressed in the power-sum basis.
    """
    lam = Partition(lam)
    mu = Partition(mu)

    Sym = SymmetricFunctions(QQ)
    s = Sym.schur()
    p = Sym.powersum()

    schur_in_powersum_basis = p(s[lam])

    return ZZ(z_mu(mu) * schur_in_powersum_basis[mu])


def isotypic_projection_of_monomial(mon, lam, R, m, n, pos,
                                    permutations, character_values):
    r"""
    Compute a nonzero scalar multiple of the lambda-isotypic
    projection of mon.

    We use the central element

        sum_{sigma in S_n} chi^lambda(sigma) sigma.

    This differs from the actual projection only by the nonzero scalar
    dim(S^lambda)/n!, which is irrelevant for ideal generation.
    """
    ans = R.zero()

    for permutation in permutations:
        mu = cycle_type_of_permutation(permutation)
        coefficient = character_values[mu]

        if coefficient != 0:
            ans += coefficient * act_on_monomial(
                mon, permutation, R, m, n, pos
            )

    return ans


# ================================================================
# Linear collision ideals
# ================================================================

def shape_of_set_partition(Pi):
    """
    Return the partition given by the block sizes of Pi.
    """
    return tuple(
        sorted([len(block) for block in Pi], reverse=True)
    )


def set_partitions_of_shape(n, shape):
    """
    Return all set partitions of [n] with the prescribed shape.
    """
    shape = tuple(shape)

    return [
        Pi for Pi in SetPartitions(n)
        if shape_of_set_partition(Pi) == shape
    ]


def linear_prime_of_set_partition(R, x, Pi, m):
    r"""
    Return the linear prime p_Pi defining the collision subspace W_Pi.
    """
    generators = []

    for block in Pi:
        block = list(block)

        for i, j in combinations(block, 2):
            for r in range(m):
                generators.append(x[r][i] - x[r][j])

    return R.ideal(generators)


def radical_specht_ideal(R, x, m, n, minimal_shapes):
    r"""
    Compute the radical ideal from Corollary 4.12:

        radical(I_lambda)
        =
        intersection_{mu in A_lambda}
        intersection_{Pi in SP_mu} p_Pi.
    """
    primes = []

    for mu in minimal_shapes:
        for Pi in set_partitions_of_shape(n, mu):
            primes.append(linear_prime_of_set_partition(R, x, Pi, m))

    assert len(primes) > 0

    J = primes[0]

    for Q in primes[1:]:
        J = J.intersection(Q)

    return R.ideal(J.groebner_basis())


# ================================================================
# Computational lemma (Lemma 5.14)
# ================================================================

def alternating_polynomial(mon, R, m, n, pos):
    r"""
    Return

        sum_{sigma in S_n} sign(sigma) sigma.mon.

    For n=3 this is epsilon_A(mon), where A is the tableau of
    shape (1,1,1).
    """
    ans = R.zero()

    for permutation in SymmetricGroup(n):
        ans += (
            permutation.sign()
            * act_on_monomial(mon, permutation, R, m, n, pos)
        )

    return ans


def B_for_111(l):
    r"""
    Construct B^(l) from Lemma 5.13 in R_{l,3}.
    """
    R, x, pos = diagonal_ring(l, 3)
    B = []

    # q_{r,s} = epsilon_A(x_{s,2} x_{r,3}), for r < s.
    for r in range(l):
        for s in range(r + 1, l):
            B.append(
                alternating_polynomial(
                    x[s][2] * x[r][3], R, l, 3, pos
                )
            )

    # p_{r,12} p_{s,13} p_{t,23}, for r <= s <= t.
    for r in range(l):
        for s in range(r, l):
            for t in range(s, l):
                p_r_12 = x[r][1] - x[r][2]
                p_s_13 = x[s][1] - x[s][3]
                p_t_23 = x[t][2] - x[t][3]

                B.append(p_r_12 * p_s_13 * p_t_23)

    return R, x, pos, B


def radical_for_111(l):
    r"""
    Compute I(V_(1,1,1)) in R_{l,3}:

        I_{12} intersection I_{13} intersection I_{23}.
    """
    R, x, pos = diagonal_ring(l, 3)

    I12 = R.ideal([x[r][1] - x[r][2] for r in range(l)])
    I13 = R.ideal([x[r][1] - x[r][3] for r in range(l)])
    I23 = R.ideal([x[r][2] - x[r][3] for r in range(l)])

    J = I12.intersection(I13).intersection(I23)

    return R.ideal(J.groebner_basis())


def verify_lemma_514():
    print("")
    print("======================================================")
    print("Verification of Lemma 5.14")
    print("======================================================")

    if USE_SINGULAR_SPOLY:
        print("S-polynomials: Singular spoly routine.")
    else:
        print("S-polynomials: explicit standard formula.")

    print("")
    print("Checking Lemma 5.14(1):")

    for l in range(1, 4):
        R, x, pos, B = B_for_111(l)

        J = radical_for_111(l)
        #print(J)
        K = R.ideal(B)
        #print(K)

        assert ideals_are_equal(J, K), \
            "Lemma 5.14(1) fails for l = %s." % l

        print("  Verified for l = %s." % l)

    print("")
    print("Checking Lemma 5.14(2):")

    for l in range(1, 7):
        R, x, pos, B = B_for_111(l)

        for f, g in combinations(B, 2):
            S = s_polynomial(f, g, R)

            assert normal_form(S, B) == 0, \
                "Lemma 5.14(2) fails for l = %s." % l

        print("  Verified for l = %s." % l)


# ================================================================
# Radicality computations 
# ================================================================

def verify_radicality_example(lam, m, n, minimal_shapes):
    r"""
    Verify I_lambda = I(V_lambda).

    The algorithm is:

      1. Compute J=I(V_lambda) as an intersection of the linear primes
         supplied by Corollary 4.12.

      2. Let D be the maximum total degree in a Groebner basis of J.

      3. Compute lambda-isotypic projections of every monomial of
         degree at most D.

      4. Let K be the ideal generated by these projections.

      5. Verify K=J.

    Since K is contained in I_lambda and I_lambda is contained in J,
    equality K=J proves I_lambda=I(V_lambda), hence radicality.
    """
    lam = Partition(lam)

    print("")
    print("======================================================")
    print("Radicality computation")
    print("lambda = %s, m = %s, n = %s" % (tuple(lam), m, n))
    print("======================================================")

    R, x, pos = diagonal_ring(m, n)

    print("")
    print("Computing I(V_lambda) as an intersection of linear primes...")

    J = radical_specht_ideal(R, x, m, n, minimal_shapes)
    GJ = list(J.groebner_basis())

    D = max(f.total_degree() for f in GJ)

    print("Number of Groebner basis elements: %s" % len(GJ))
    print("Largest total degree in this Groebner basis: D = %s" % D)

    Sn = SymmetricGroup(n)
    permutations = list(Sn)

    conjugacy_classes = sorted(
        set(
            cycle_type_of_permutation(permutation)
            for permutation in permutations
        ),
        reverse=True
    )

    character_values = {
        mu: character_value(lam, mu)
        for mu in conjugacy_classes
    }

    print("")
    print("Character values computed by Sage:")

    for mu in conjugacy_classes:
        print(
            "  chi^%s(%s) = %s"
            % (tuple(lam), tuple(mu), character_values[mu])
        )

    print("")
    print("Generating all monomials of degree at most D...")

    monomials = monomials_of_degree_at_most(R, D)

    print("Number of monomials: %s" % len(monomials))

    print("")
    print("Computing lambda-isotypic projections...")

    projected_generators = []

    for mon in monomials:
        projection = isotypic_projection_of_monomial(
            mon,
            lam,
            R,
            m,
            n,
            pos,
            permutations,
            character_values
        )

        if projection != 0:
            projected_generators.append(projection)

    print(
        "Number of nonzero projected generators: %s"
        % len(projected_generators)
    )

    K = R.ideal(projected_generators)

    print("")
    print("Testing equality with I(V_lambda)...")

    assert ideals_are_equal(K, J), \
        "The projected ideal is not equal to I(V_lambda)."

    print("")
    print("Success.")
    print(
        "The lambda-isotypic projections of all monomials of total"
    )
    print(
        "degree at most %s generate I(V_lambda)." % D
    )
    print(
        "Therefore I_%s is radical in R_{%s,%s}."
        % (tuple(lam), m, n)
    )

    return R, J, K, D


# ================================================================
# Run all computations
# ================================================================

if __name__ == "__main__":

# ------------------------------------------------------------
# Radicality
#
# Verify if the ideal I_\lambda is radical or not
# The results of the computations are stored in the file given as output
# Change lam, m, n, and minimal_shapes depending on the ideal one wants to check
# lam is the partition lambda of n, m is the number of sets of variables,
# minimal_shapes is the set of minimal shapes that are not dominated by lambda
# and minimal for the refining order.
#
# It gives either:
#   - a positive result (the ideal is radical)
#   - an error in assert ideals_are_equal(K,J) (the ideal is not radical)
#   - it aborts when the computation is to big.
# ------------------------------------------------------------

     from contextlib import redirect_stdout
     with open("hook_11111_output.txt", "w") as output:
         with redirect_stdout(output):
             verify_radicality_example(
                 lam=(1,1,1,1,1),
                 m=2,
                 n=5,
                 minimal_shapes=[(2,1,1,1)]
             )


#-------------------------------------------------------------
# Verifications of Lemma 5.14
#     - (1) verifies that a basis of I(V_{(1,1,1)}) is as given in the lemma for 1<=l<=3
#     - (2) verifies that the set of polynomials is a Grobner basis
#-------------------------------------------------------------
             
     verify_lemma_514()
