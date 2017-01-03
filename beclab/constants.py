import numpy


# Physical constants

#: Bohr radius (in m).
R_BOHR = 5.2917720859e-11
#: Planck constant (in J*s).
HBAR = 1.054571628e-34
#: Proton mass (in kg).
M_PROTON = 1.6588053425287357e-27


class Component:
    """
    Represents a BEC component.

    .. py:attribute:: atom_name

        A string with the type of the atom.

    .. py:attribute:: atom_mass

        Atom mass (in a.m.u).

    .. py:attribute:: f

        Total spin of the atom.

    .. py:attribute:: mf

        Total spin projection.

    .. py:attribute:: m

        Atom mass (in kg).
    """

    def __init__(self, atom_name, atom_mass, f, mf):
        self.atom_name = atom_name
        self.atom_mass = atom_mass
        self.m = M_PROTON * atom_mass
        self.f = f
        self.mf = mf

    def __repr__(self):
        return "Component(" + \
            ",".join([str(x) for x in (self.atom_name, self.atom_mass, self.f, self.mf)]) + ")"


class FeshbachResonance:
    """
    Represents a Feshbach resonance.

    .. py:attribute:: value

        Resonance magnetic field (in G).

    .. py:attribute:: background_scattering

        Background scattering length (in Bohr radii).

    .. py:attribute:: width

        Resonance width (in G).

    .. py:attribute:: decay width

        Resonance decay width (in G).
    """

    def __init__(self, value, background_scattering, width, decay_width):
        self.value = value
        self.background_scattering = background_scattering
        self.width = width
        self.decay_width = decay_width


#: :math:`^{87}\mathrm{Rb}`, :math:`\vert F=1,\, m_F=-1 \rangle`.
rb87_1_minus1 = Component("Rb", 87, 1, -1)
#: :math:`^{87}\mathrm{Rb}`, :math:`\vert F=1,\, m_F=+1 \rangle`.
rb87_1_1 = Component("Rb", 87, 1, 1)
#: :math:`^{87}\mathrm{Rb}`, :math:`\vert F=2,\, m_F=-1 \rangle`.
rb87_2_minus1 = Component("Rb", 87, 2, -1)
#: :math:`^{87}\mathrm{Rb}`, :math:`\vert F=2,\, m_F=-1 \rangle`.
rb87_2_1 = Component("Rb", 87, 2, 1)
#: :math:`^{87}\mathrm{Rb}`, :math:`\vert F=2,\, m_F=+2 \rangle`.
rb87_2_2 = Component("Rb", 87, 2, 2)

#: The "magical" magnetic field (in G)
#: for :math:`^{87}\mathrm{Rb}`, :math:`\vert F=1,\, m_F=-1 \rangle` and
#: :math:`^{87}\mathrm{Rb}`, :math:`\vert F=2,\, m_F=+1 \rangle` interaction.
magical_field_Rb87_1m1_2p1 = 3.228

#: The Feshbach resonance for Rb87 at 9.1G.
#: See Kaufman et al., PRA 80 050701 (2009).
resonance_Rb_9_1 = FeshbachResonance(9.1047, 97.7, 2e-3, 4.7e-3)


def feshbach_interaction(m, B, resonance):
    r"""
    Returns a tuple of the scattering length (in Bohr radii)
    and the corresponding two-body loss rate :math:`\kappa`
    (in :math:`\mathrm{m}^3/\mathrm{s}`)
    for the given Feshbach resonance.

    :param m: atom mass (kg).
    :param B: magnetic field (G).
    :param resonance: a :py:class:`FeshbachResonance` object.
    """
    a_bg = resonance.background_scattering * R_BOHR
    denom = (B - resonance.value) ** 2 + resonance.decay_width ** 2 / 4
    a12 = a_bg * (1 - resonance.width * (B - resonance.value) / denom)
    gamma12 = 4 * numpy.pi * HBAR * (a_bg / 2 * resonance.width * resonance.decay_width / denom) / m
    kappa12 = gamma12 / 2
    return a12 / R_BOHR, kappa12


def scattering_length(comp1, comp2, B=None):
    """
    Returns the scattering length (in Bohr radii) for two :py:class:`Component` objects
    in the magnetic field ``B`` (in G).
    """
    comp1, comp2 = sorted([comp1, comp2], key=lambda c: (c.f, c.mf))

    # Currently only Rb87 is supported
    for c in (comp1, comp2):
        if not (c.atom_name == "Rb" and c.atom_mass == 87):
            raise NotImplementedError

    if comp1.f == comp2.f == 1:
        return 100.4

    # Egorov et al., arXiv:1204.1591 (2012)
    if (B == magical_field_Rb87_1m1_2p1
            and (comp1.f, comp1.mf) == (1, -1) and (comp2.f, comp2.mf) == (2, 1)):
        return 98.006
    if (B == magical_field_Rb87_1m1_2p1
            and (comp1.f, comp1.mf) == (2, 1) and (comp2.f, comp2.mf) == (2, 1)):
        return 95.44

    if ((comp1.f, comp1.mf) == (1, 1) and (comp2.f, comp2.mf) == (2, -1) and
            abs(B - resonance_Rb_9_1.value) < 1):
        a12, _ = feshbach_interaction(comp1.m, B, resonance_Rb_9_1)
        return a12

    # TODO: check that it's actually correct
    if (comp1.f, comp1.mf) == (2, -1) and (comp2.f, comp2.mf) == (2, -1):
        return 95.44

    # Tojo et al., PRA 80 042704 (2009)
    if (comp1.f, comp1.mf) == (2, 2) and (comp2.f, comp2.mf) == (2, 2):
        return 95.68

    raise NotImplementedError(
        "Unknown B: " + str(B) +
        " and components: " + repr(tuple([comp1, comp2])))


def loss_rate(*comps, **kwds):
    r"""
    Returns the loss rate (in SI units) for a list of one or more :py:class:`Component` objects
    in the magnetic field ``B`` (in G).
    The loss rate returned is the theoretical one (:math:`\kappa`).
    """

    assert list(kwds.keys()) == ['B']
    B = kwds['B']
    comps = sorted(comps, key=lambda c: (c.f, c.mf))

    # Currently only Rb87 is supported
    for c in comps:
        if not (c.atom_name == "Rb" and c.atom_mass == 87):
            raise NotImplementedError

    if len(comps) == 2 and comps[0].f == comps[1].f == 1:
        return 0.

    # 1-1-1 loss
    if len(comps) == 3 and comps[0].f == comps[1].f == comps[2].f == 1:
        return 5.4e-42 / 6

    # Egorov et al., arXiv:1204.1591 (2012)
    if (B == magical_field_Rb87_1m1_2p1 and len(comps) == 2
            and (comps[0].f, comps[0].mf) == (1, -1)
            and (comps[1].f, comps[1].mf) == (2, 1)):
        return 1.51e-20 / 2
    if (B == magical_field_Rb87_1m1_2p1 and len(comps) == 2
            and (comps[0].f, comps[0].mf) == (2, 1)
            and (comps[1].f, comps[1].mf) == (2, 1)):
        return 8.1e-20 / 4

    # Kaufman et al., PRA 80 050701 (2009)
    if len(comps) == 2 and (comps[0].f, comps[0].mf) == (1, 1) and \
            (comps[1].f, comps[1].mf) == (2, -1) and abs(B - resonance_Rb_9_1.value) < 1:
        _, kappa12 = feshbach_interaction(comps[0].m, B, resonance_Rb_9_1)
        return kappa12

    # TODO: check that it's actually correct
    if (len(comps) == 2 and (comps[0].f, comps[0].mf) == (2, -1)
            and (comps[1].f, comps[1].mf) == (2, -1)):
        return 8.1e-20 / 4

    # Tojo et al., PRA 80 042704 (2009)
    if len(comps) == 2 and (comps[0].f, comps[0].mf) == (2, 2) and \
            (comps[1].f, comps[1].mf) == (2, 2):
        return 0. # TODO: check this

    raise NotImplementedError(
        "Unknown B: " + str(B) +
        " and components: " + repr(comps))


def effective_area(m, f1, f2):
    """
    Returns the effective area (in squared meters) for a pseudo-1D trap
    with transverse frequencies ``f1`` and ``f2`` (in Hz) and atoms with mass ``m`` (in kg).
    """
    eff_length = lambda f: numpy.sqrt(HBAR / (m * 2 * numpy.pi * f))
    return 2.0 * numpy.pi * eff_length(f1) * eff_length(f2)


def scattering_3d(a, m):
    r"""
    Returns the 3D elastic interaction coefficient
    (in :math:`\mathrm{J} \cdot \mathrm{m}^3`) for
    the scattering length ``a`` (in Bohr radii) and atoms with mass ``m`` (in kg).
    """
    return 4 * numpy.pi * HBAR ** 2 * a * R_BOHR / m


def scattering_matrix(comps, B=None):
    """
    Returns a numpy array of the elastic interaction coefficients for
    the given list of :py:class:`Component` objects and magnetic field ``B`` (in G).
    """
    get_g = lambda c1, c2: scattering_3d(scattering_length(c1, c2, B=B), c1.m)
    return numpy.array([[get_g(c1, c2) for c2 in comps] for c1 in comps])


def mu_tf_3d(freqs, N, m, g):
    """
    Returns the TF-approximated chemical potential (in J) for ``N`` atoms of mass ``m``
    and elastic interaction coefficient ``g`` (in :math:`\mathrm{J} \cdot \mathrm{m}^3`)
    in a 3D trap with frequencies ``freqs`` (in Hz).
    """
    assert len(freqs) == 3
    ws = [2 * numpy.pi * f for f in freqs]
    w = (ws[0] * ws[1] * ws[2]) ** (1. / 3)
    return (
        (15 * N / (8.0 * numpy.pi)) ** 0.4 *
        (m * w ** 2 / 2) ** 0.6 *
        g ** 0.4)


def mu_tf_1d(freqs, N, m, g):
    """
    Returns the TF-approximated chemical potential (in J) for ``N`` atoms of mass ``m``
    and elastic interaction coefficient ``g`` (in :math:`\mathrm{J} \cdot \mathrm{m}`)
    in a 1D trap with a 1-element array of frequencies ``freqs`` (in Hz).
    """
    assert len(freqs) == 1
    return (
        (0.75 * g * N) ** (2. / 3) *
        (m * (2 * numpy.pi * freqs[0]) ** 2 / 2) ** (1. / 3))
