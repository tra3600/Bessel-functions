"""Tests : chaque implémentation « maison » est comparée à scipy ou à une identité exacte."""

import numpy as np
import pytest
from scipy import special

import bessel as B
import quiz
from BesselFunctions import CAS, Afficheur, lancer, main

X = np.linspace(0.3, 20, 60)


@pytest.mark.parametrize("nu", [0, 1, 2, 0.5, 2.7, 7])
def test_serie_J(nu):
    assert np.allclose(B.serie_J(nu, X), special.jv(nu, X), atol=1e-8)


@pytest.mark.parametrize("nu", [0, 1, 3.3])
def test_serie_I(nu):
    x = np.linspace(0.1, 10, 40)
    assert np.allclose(B.serie_I(nu, x), special.iv(nu, x), rtol=1e-12)


@pytest.mark.parametrize("n", [0, 1, 4, 9])
def test_integrale(n):
    assert np.allclose(B.integrale_J(n, np.linspace(0, 60, 40)),
                       special.jv(n, np.linspace(0, 60, 40)), atol=1e-12)


@pytest.mark.parametrize("x", [0.5, 3.0, 10.0, 40.0])
def test_miller(x):
    n = np.arange(31)
    assert np.allclose(B.miller(x, 30), special.jv(n, x), atol=1e-13)


def test_miller_zero():
    assert list(B.miller(0.0, 3)) == [1.0, 0.0, 0.0, 0.0]


def test_montante_instable():
    ex = special.jv(np.arange(31), 10.0)
    assert abs(B.recurrence_montante(10.0, 30)[30] / ex[30] - 1) > 1


@pytest.mark.parametrize("nu", [0, 1, 2.5, 6])
def test_equation_et_recurrences(nu):
    x = np.linspace(0.5, 20, 50)
    assert np.max(np.abs(B.residu_equation(nu, x))) < 1e-8
    assert np.max(np.abs(B.residu_recurrence(nu, x))) < 1e-12
    assert np.max(np.abs(B.residu_derivee(nu, x))) < 1e-12
    assert np.allclose(B.wronskien(nu, x), 2 / (np.pi * x))
    assert np.max(np.abs(B.residu_equation_modifiee(nu, x[:30]))) < 1e-8


def test_identites():
    assert np.allclose(B.somme_carres(np.array([0.1, 5.0, 30.0])), 1.0)
    assert np.isclose(B.fonction_generatrice(3.0, 1.7), np.exp(1.5 * (1.7 - 1 / 1.7)))
    th = np.linspace(0, 6, 9)
    assert np.allclose(B.jacobi_anger(4.0, th), np.exp(4j * np.sin(th)))


def test_asymptotique():
    assert np.isclose(B.asymptotique_J(0, 50.0, 4), special.j0(50.0), atol=1e-12)
    assert np.isclose(B.asymptotique_J(2, 30.0, 4), special.jv(2, 30.0), atol=1e-9)
    assert np.isclose(B.asymptotique_I(0, 60.0), special.iv(0, 60.0), rtol=1e-3)
    assert np.isclose(B.asymptotique_K(0, 60.0), special.kv(0, 60.0), rtol=1e-3)
    assert np.all(np.abs(special.jv(3, X + 5)) <= B.enveloppe(X + 5) * 1.5)


@pytest.mark.parametrize("nu", [0, 1, 2, 5])
def test_zeros(nu):
    z = B.zeros_J(nu, 6)
    assert np.allclose(special.jv(nu, z), 0, atol=1e-12)
    assert np.allclose(B.mcmahon(nu, np.arange(1, 7))[2:], z[2:], atol=2e-2 * (nu + 1))
    assert np.isclose(B.newton_zero(nu, z[1] + 0.05), z[1])
    zp = B.zeros_derivee(nu, 3)
    assert np.allclose(special.jvp(nu, zp), 0, atol=1e-12)


def test_entrelacement():
    z0, z1 = B.zeros_J(0, 10), B.zeros_J(1, 10)
    assert np.all(z0 < z1) and np.all(z1[:-1] < z0[1:])


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_spheriques(n):
    x = np.linspace(0.5, 30, 50)
    assert np.allclose(B.sph_j_fermee(n, x), special.spherical_jn(n, x), atol=1e-11)
    assert np.allclose(B.sph_j_depuis_J(n, x), special.spherical_jn(n, x), atol=1e-12)
    assert np.allclose(B.sph_j_recurrence(5, 9.0), special.spherical_jn(np.arange(6), 9.0))


def test_tambour():
    r = np.linspace(0, 1, 5)
    u, alpha = B.mode_tambour(2, 1, r, 0.0)
    assert np.isclose(u[-1], 0, atol=1e-12) and np.isclose(alpha, 5.1356223)
    assert np.isclose(B.pendule_corde_pesante(k=2)[1], 5.5200781 / 2.4048256)


def test_fourier_bessel():
    alphas, c = B.coefficients_fourier_bessel(lambda r: 1 - r ** 2, 0, 12)
    r = np.linspace(0, 1, 50)
    approx = B.chaleur_cylindre(alphas, c, r, 0.0)
    assert np.max(np.abs(approx - (1 - r ** 2))) < 1e-2
    assert B.chaleur_cylindre(alphas, c, 0.0, 1.0) < 1e-2       # ~ c1 e^(-5.78)


def test_airy():
    assert B.airy(np.array([0.0]))[0] == 1.0
    assert np.isclose(B.airy(np.array([special.jn_zeros(1, 1)[0]]))[0], 0, atol=1e-25)


def test_fm_spectre():
    fs = 8192
    t = np.arange(fs) / fs
    spec = np.abs(np.fft.rfft(B.signal_fm(2.0, 1000, 100, t))) * 2 / fs
    for n in range(4):
        assert np.isclose(spec[1000 + 100 * n], abs(special.jv(n, 2.0)), atol=1e-8)


@pytest.mark.parametrize("e", [0.1, 0.5, 0.8])
def test_kepler(e):
    M = np.linspace(0, 2 * np.pi, 40)
    E = B.kepler_newton(M, e)
    assert np.allclose(E - e * np.sin(E), M)
    assert np.allclose(B.kepler_bessel(M, e, 200), E, atol=1e-6)


def test_effet_de_peau():
    r = np.linspace(0, 1, 5)
    bas, haut = B.effet_de_peau(r, 1), B.effet_de_peau(r, 500)
    assert np.isclose(bas[-1], 1) and haut[0] < 1e-3 < bas[0]


# ----- quiz et interface -------------------------------------------------
def test_quiz_coherent():
    qs = quiz.generer(45, graine=1)
    assert len(qs) == 45
    for enonce, choix, bonne, expl in qs:
        assert bonne in choix and len(set(choix)) == len(choix) >= 2 and expl


def test_quiz_reproductible():
    assert quiz.generer(9, graine=3) == quiz.generer(9, graine=3)


def test_quiz_jouer_parfait():
    qs = quiz.generer(6, graine=5)
    reponses = iter("ABCD"[c.index(b)] for _, c, b, _ in qs)
    assert quiz.jouer(6, 5, lire=lambda _: next(reponses), ecrire=lambda *a: None) == 6


def test_quiz_fin_entree():
    def eof(_):
        raise EOFError
    assert quiz.jouer(3, 1, lire=eof, ecrire=lambda *a: None) == 0


@pytest.mark.parametrize("nom", list(CAS))
def test_chaque_cas_sans_graphique(nom, capsys):
    lancer(nom, Afficheur(graphique=False))
    assert capsys.readouterr().out.strip()


def test_cli(tmp_path, capsys):
    assert main(["--liste"]) == 0
    assert main(["--cas", "tambour", "--sauver", str(tmp_path)]) == 0
    assert (tmp_path / "tambour.png").exists()
    assert main(["--cas", "diffraction", "--sauver", str(tmp_path)]) == 0
    assert (tmp_path / "airy.png").exists()
