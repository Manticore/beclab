import numpy


# Physical constants
R_BOHR = 5.2917720859e-11 # Bohr radius, in m
HBAR = 1.054571628e-34 # Planck constant, in J*s
M_PROTON = 1.6588053425287357e-27 # mass of a proton, in kg


class State:

    def __init__(self, atom_name, atom_mass, f, mf):
        self.atom_name = atom_name
        self.atom_mass = atom_mass
        self.m = M_PROTON * atom_mass
        self.f = f
        self.mf = mf

    def __repr__(self):
        return "State(" + \
            ",".join([str(x) for x in (self.atom_name, self.atom_mass, self.f, self.mf)]) + ")"


Rb87_1_minus1 = State("Rb", 87, 1, -1)
Rb87_1_1 = State("Rb", 87, 1, 1)
Rb87_2_minus1 = State("Rb", 87, 2, -1)
Rb87_2_1 = State("Rb", 87, 2, 1)
Rb87_2_2 = State("Rb", 87, 2, 2)

# "Magical" B for (1,-1)-(2,1) interaction
B_Rb87_1m1_2p1 = 3.228


def get_feshbach_interaction(m, B, B0, a_bg, delta, gamma):
    """
    Returns a tuple of the scattering length (in Bohr radii)
    and the corresponding two-body loss rate (in m^3/s) for the given Feshbach resonance:
    m: atom mass (kg)
    B: magnetic field (G)
    B0: resonance value (G)
    a_bg: background scattering length (Bohr radii)
    delta: resonance width (G)
    gamma: decay width (G)
    """
    a_bg *= R_BOHR
    denom = (B - B0) ** 2 + gamma ** 2 / 4
    a12 = a_bg * (1 - delta * (B - B0) / denom)
    gamma12 = 4 * numpy.pi * HBAR * (a_bg / 2 * delta * gamma / denom) / m
    return a12 / R_BOHR, gamma12


def get_interaction_constants(B, *states):
    """
    Returns a tuple of the scattering length (for ``len(states) == 2``, otherwise 0) in Bohr radii
    and the loss rate in SI units.
    """

    # Currently only Rb87 is supported
    for c in states:
        assert c.atom_name == "Rb" and c.atom_mass == 87

    # sort by f,mf to avoid multiple dispatch
    states = sorted(states, key=lambda c: (c.f, c.mf))

    # 1-1 interaction
    if len(states) == 2 and states[0].f == states[1].f == 1:
        return 100.4, 0.

    # 1-1-1 loss
    if len(states) == 3 and states[0].f == states[1].f == states[2].f == 1:
        return 0., 5.4e-42

    # Egorov et al., arXiv:1204.1591 (2012)
    if (B == B_Rb87_1m1_2p1 and len(states) == 2
            and (states[0].f, states[0].mf) == (1, -1)
            and (states[1].f, states[1].mf) == (2, 1)):
        return 98.006, 1.51e-20
    if (B == B_Rb87_1m1_2p1 and len(states) == 2
            and (states[0].f, states[0].mf) == (2, 1)
            and (states[1].f, states[1].mf) == (2, 1)):
        return 95.44, 8.1e-20

    # Kaufman et al., PRA 80 050701 (2009)
    if len(states) == 2 and (states[0].f, states[0].mf) == (1, 1) and \
            (states[1].f, states[1].mf) == (2, -1) and abs(B - B0) < 1:
        a_bg = 97.7 # background scattering length, Bohr radii
        gamma = 4.7e-3 # decay rate, G
        delta = 2e-3 # width of the resonance, G
        B0 = 9.105 # resonance
        a12, gamma12 = get_feshbach_interaction(m, B, B0, a_bg, delta, gamma)
        return a12, gamma12

    # TODO: check that it's actually correct
    if (len(states) == 2 and (states[0].f, states[0].mf) == (2, -1)
            and (states[1].f, states[1].mf) == (2, -1)):
        return 95.44, 8.1e-20

    # Tojo et al., PRA 80 042704 (2009)
    if len(states) == 2 and (states[0].f, states[0].mf) == (2, 2) and \
            (states[1].f, states[1].mf) == (2, 2) and abs(B - B0) < 1:
        return 95.68, 0. # TODO: check this

    raise NotImplementedError("Unknown B " + str(B) + " and states: " + repr(states))


def get_effective_area(m, f1, f2):
    eff_length = lambda f:  numpy.sqrt(HBAR / (m * 2 * numpy.pi * f))
    return 2.0 * numpy.pi * eff_length(f1) * eff_length(f2)


def get_scattering_constant(a, m):
    return 4 * numpy.pi * HBAR ** 2 * a * R_BOHR / m


def muTF3D(freqs, N, state):
    """
    Returns the TF-approximated chemical potential, in Joules.
    freqs: trap frequencies (Hz)
    N: atom number
    state: a State object
    """
    assert len(freqs) == 3
    ws = [2 * numpy.pi * f for f in freqs]
    a, _ = get_interaction_constants(None, state, state)
    g_3D = get_scattering_constant(a, state.m)

    if len(freqs) == 3:
        w = (ws[0] * ws[1] * ws[2]) ** (1. / 3)
        return (
            (15 * N / (8.0 * numpy.pi)) ** 0.4 *
            (state.m * w ** 2 / 2) ** 0.6 *
            g_3D ** 0.4)

def muTF1D(freqs, N, state, S_eff):
    a, _ = get_interaction_constants(None, state, state)
    g_3D = get_scattering_constant(a, state.m)
    g_1D = g_3D / S_eff
    return (
        (0.75 * g_1D * N) ** (2. / 3) *
        (state.m * (2 * numpy.pi * freqs[0]) ** 2 / 2) ** (1. / 3))
