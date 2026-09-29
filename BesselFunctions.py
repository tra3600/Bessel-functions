"""Le laboratoire de Bessel : des cas illustrés, calculés, et un quiz.

Les fonctions de Bessel apparaissent dès qu'il y a une symétrie circulaire ou
cylindrique : tambour, chaleur dans un cylindre, diffraction, orbites de
Kepler, synthèse sonore FM...

Exemples :
    python BesselFunctions.py                       # menu interactif
    python BesselFunctions.py --cas tambour         # un cas précis (répétable)
    python BesselFunctions.py --tout                # tous les cas, figures à l'écran
    python BesselFunctions.py --tout --sauver figures
    python BesselFunctions.py --tout --sans-graphique
    python BesselFunctions.py --quiz 10 --graine 42
    python BesselFunctions.py --liste
"""

import argparse
import os
import sys

import numpy as np
from scipy import special

import bessel as B

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ----------------------------------------------------------------------
# Affichage
# ----------------------------------------------------------------------
class Afficheur:
    """Gère les figures : à l'écran, en PNG, ou pas du tout."""

    def __init__(self, graphique=True, dossier=None):
        self.graphique = graphique or dossier is not None
        self.dossier = dossier
        if dossier:
            import matplotlib
            matplotlib.use("Agg")
            os.makedirs(dossier, exist_ok=True)

    def __call__(self, nom, **kw):
        if not self.graphique:
            return
        import matplotlib.pyplot as plt
        import illustrations as ill
        fig = getattr(ill, f"fig_{nom}")(**kw)
        if self.dossier:
            chemin = os.path.join(self.dossier, f"{nom}.png")
            fig.savefig(chemin, dpi=130)
            print(f"   🖼  figure enregistrée : {chemin}")
            plt.close(fig)
        else:
            plt.show()


def titre(texte):
    print("\n" + "═" * 72)
    print(f"  {texte}")
    print("═" * 72)


def para(texte):
    print(texte.strip("\n"))


def maxerr(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ----------------------------------------------------------------------
# Les cas illustrés
# ----------------------------------------------------------------------
def cas_premieres(montrer):
    titre("1. LES PREMIÈRES FONCTIONS : J_n et Y_n")
    para("""
Les fonctions de Bessel résolvent  x² y'' + x y' + (x² − n²) y = 0.
Deux solutions indépendantes : J_n (première espèce, sage en 0) et Y_n
(deuxième espèce, qui explose en 0). Comme cos et sin, elles oscillent...
mais avec une amplitude qui décroît en 1/√x.
""")
    x = np.array([0.0, 1.0, 5.0, 10.0])
    print(f"{'x':>6}" + "".join(f"{'J_' + str(n) + '(x)':>12}" for n in range(4)))
    for xv in x:
        print(f"{xv:6.1f}" + "".join(f"{special.jv(n, xv):12.6f}" for n in range(4)))
    print("\nJ_0(0) = 1 et J_n(0) = 0 pour n ≥ 1 ; Y_n(x) → −∞ quand x → 0 :")
    print("   Y_0(0.01) =", f"{special.y0(0.01):.4f}", "  Y_1(0.01) =", f"{special.y1(0.01):.1f}")
    print("\nParité : J_n(−x) = (−1)^n J_n(x)   et   J_(−n) = (−1)^n J_n")
    print(f"   J_3(−2) = {special.jv(3, -2.0):+.6f} = −J_3(2) = {-special.jv(3, 2.0):+.6f}")
    print(f"   J_(−3)(2) = {special.jv(-3, 2.0):+.6f} = −J_3(2)")
    montrer("premieres")


def cas_equation(montrer):
    titre("2. L'ÉQUATION DE BESSEL : vérifiée numériquement")
    para("""
On ne se contente pas de l'affirmer : on calcule le résidu
x² y'' + x y' + (x² − ν²) y  pour ν entier, demi-entier, quelconque, et pour
la solution J_ν comme pour I_ν (équation modifiée, signe − devant x²).
""")
    x = np.linspace(0.5, 20, 200)
    for nu in (0, 1, 2.5, 0.3, 7):
        print(f"   ν = {nu:<4}  résidu max sur [0.5, 20] : {maxerr(B.residu_equation(nu, x), 0):.2e}")
    print("\nÉquation modifiée avec I_ν :")
    for nu in (0, 1, 2.5):
        r = B.residu_equation_modifiee(nu, x[:60])
        print(f"   ν = {nu:<4}  résidu max sur [0.5, 6.4]  : {maxerr(r, 0):.2e}")
    print("\nWronskien  J_(ν+1) Y_ν − J_ν Y_(ν+1) = 2/(π x)  (indépendance des solutions) :")
    for nu in (0, 1, 3.5):
        w = B.wronskien(nu, x)
        print(f"   ν = {nu:<4}  écart à 2/(πx) : {maxerr(w, 2 / (np.pi * x)):.2e}")


def cas_serie(montrer):
    titre("3. LA SÉRIE ENTIÈRE : belle, mais traître")
    para("""
J_ν(x) = Σ (−1)^k / (k! Γ(k+ν+1)) · (x/2)^(2k+ν)  converge pour tout x.
Mais pour x grand, les termes montent jusqu'à ~e^x avant de s'annuler : on perd
tous les chiffres par « annulation catastrophique ».
""")
    print(f"{'x':>5}{'erreur série':>16}{'chiffres perdus':>18}")
    for xv in (1, 5, 10, 20, 30, 40):
        s = float(B.serie_J(0, np.array(float(xv))))
        e = abs(s - special.j0(xv))
        print(f"{xv:5d}{e:16.2e}{max(0, 16 + np.log10(e + 1e-17)):18.0f}")
    print("\nMême série, ordre réel non entier (Γ généralise la factorielle) :")
    for nu in (0.5, 1 / 3, 2.7):
        print(f"   J_{nu:.3f}(3) série = {float(B.serie_J(nu, np.array(3.0))):.10f}"
              f"   scipy = {special.jv(nu, 3.0):.10f}")
    montrer("serie")


def cas_integrale(montrer):
    titre("4. L'INTÉGRALE DE BESSEL, LA GÉNÉRATRICE, JACOBI–ANGER")
    para("""
Pour n entier :   J_n(x) = (1/2π) ∫ cos(nτ − x sin τ) dτ.
Méthode des trapèzes sur une fonction périodique = convergence exponentielle :
256 points donnent la précision machine, même à x = 50, là où la série échoue.
""")
    for n, xv in [(0, 1.0), (3, 10.0), (5, 50.0)]:
        val = float(B.integrale_J(n, np.array(xv)))
        print(f"   J_{n}({xv:>4}) intégrale = {val:+.12f}   scipy = {special.jv(n, xv):+.12f}")
    para("""
Fonction génératrice :  exp( x/2 · (t − 1/t) ) = Σ_{n∈Z} J_n(x) tⁿ.
En posant t = e^{iθ} on retrouve les séries de Fourier (Jacobi–Anger) :
e^{i x sin θ} = Σ J_n(x) e^{inθ}.
""")
    x, t = 3.0, 1.7
    print(f"   somme  = {B.fonction_generatrice(x, t):.12f}")
    print(f"   exp(...) = {np.exp(x / 2 * (t - 1 / t)):.12f}")
    th = np.linspace(0, 2 * np.pi, 7)
    err = maxerr(B.jacobi_anger(x, th), np.exp(1j * x * np.sin(th)))
    print(f"   Jacobi–Anger, écart max : {err:.2e}")
    xs = np.array([0.3, 4.0, 17.0])
    print(f"   J_0² + 2ΣJ_n² (doit valoir 1) : {B.somme_carres(xs)}")


def cas_recurrences(montrer):
    titre("5. RÉCURRENCES : la montante échoue, celle de Miller réussit")
    para("""
J_(n−1) + J_(n+1) = (2n/x) J_n   et   J'_n = J_(n−1) − (n/x) J_n.
Idée naïve : partir de J_0, J_1 et monter. Elle est instable pour n > x (on
suit la solution Y_n qui explose). Remède (Miller) : DESCENDRE depuis un rang
très élevé avec des valeurs bidon, puis normaliser par 1 = J_0 + 2ΣJ_(2k).
""")
    x = np.array([0.7, 3.0, 12.5])
    for nu in (1, 2, 4.5):
        print(f"   ν = {nu:<4} résidu récurrence : {maxerr(B.residu_recurrence(nu, x), 0):.1e}"
              f"   résidu dérivée : {maxerr(B.residu_derivee(nu, x), 0):.1e}")
    xv, N = 10.0, 30
    ex = special.jv(np.arange(N + 1), xv)
    print(f"\n   J_n({xv:g}) pour n = 0..{N} :")
    print(f"   montante  : erreur max {maxerr(B.recurrence_montante(xv, N), ex):.2e}"
          f"  (rel. sur J_{N} : {abs(B.recurrence_montante(xv, N)[N] / ex[N] - 1):.1e})")
    print(f"   Miller    : erreur max {maxerr(B.miller(xv, N), ex):.2e}"
          f"  (rel. sur J_{N} : {abs(B.miller(xv, N)[N] / ex[N] - 1):.1e})")
    montrer("recurrences")


def cas_asymptotique(montrer):
    titre("6. GRANDS x : l'approximation de Hankel")
    para("""
Pour x grand :  J_n(x) ≈ √(2/πx) · cos(x − nπ/2 − π/4)  (+ corrections en 1/x).
Bessel ≈ cosinus « amorti » : un déphasage de nπ/2 + π/4 et une enveloppe en 1/√x.
""")
    print(f"{'x':>5}{'J_0 exacte':>14}{'1 terme':>14}{'4 termes':>14}{'erreur (4)':>13}")
    for xv in (2.0, 5.0, 10.0, 30.0, 100.0):
        e = special.j0(xv)
        a1 = float(B.asymptotique_J(0, np.array(xv), 1))
        a4 = float(B.asymptotique_J(0, np.array(xv), 4))
        print(f"{xv:5.0f}{e:14.8f}{a1:14.8f}{a4:14.8f}{abs(a4 - e):13.1e}")
    xv = 30.0
    print("\nFonctions modifiées à x = 30 (approximations exponentielles) :")
    print(f"   I_1 exacte {special.iv(1, xv):.6e}  asymptotique {float(B.asymptotique_I(1, np.array(xv))):.6e}")
    print(f"   K_1 exacte {special.kv(1, xv):.6e}  asymptotique {float(B.asymptotique_K(1, np.array(xv))):.6e}")
    montrer("asymptotique")


def cas_modifiees(montrer):
    titre("7. FONCTIONS MODIFIÉES : I_n et K_n (exponentielles, sans oscillation)")
    para("""
Changer x en ix : I_n(x) = i^(−n) J_n(ix). Les oscillations deviennent des
exponentielles : I_n croît comme e^x/√(2πx), K_n décroît comme √(π/2x)·e^(−x).
Elles décrivent la diffusion, l'écrantage, les ondes évanescentes.
""")
    x = np.array([0.5, 2.0, 5.0])
    for n in (0, 1):
        print(f"   I_{n}(x) = " + "  ".join(f"{v:.6f}" for v in special.iv(n, x))
              + f"   série : {maxerr(B.serie_I(n, x), special.iv(n, x)):.1e}")
    for n in (0, 1):
        print(f"   K_{n}(x) = " + "  ".join(f"{v:.6f}" for v in special.kv(n, x)))
    w = 5.0
    print(f"\nWronskien : I_ν K_(ν+1) + I_(ν+1) K_ν = 1/x → "
          f"{special.iv(1, w) * special.kv(2, w) + special.iv(2, w) * special.kv(1, w):.8f} = {1 / w:.8f}")
    print(f"Lien complexe : I_0(2) = J_0(2i) → {special.jv(0, 2j).real:.8f} = {special.iv(0, 2.0):.8f}")
    montrer("modifiees")


def cas_zeros(montrer):
    titre("8. LES ZÉROS : entrelacés et presque équidistants")
    para("""
J_n a une infinité de zéros j_(n,k) ; ceux de J_n et J_(n+1) s'entrelacent.
Écart asymptotique → π. Approximation de McMahon, puis raffinement par Newton.
""")
    print(f"{'k':>3}{'j_0,k':>11}{'j_1,k':>11}{'j_2,k':>11}{'écart j_0':>11}{'McMahon j_0':>13}")
    z = [B.zeros_J(n, 6) for n in range(3)]
    mc = B.mcmahon(0, np.arange(1, 7))
    for k in range(6):
        ec = z[0][k] - z[0][k - 1] if k else float("nan")
        print(f"{k + 1:3d}{z[0][k]:11.5f}{z[1][k]:11.5f}{z[2][k]:11.5f}{ec:11.5f}{mc[k]:13.5f}")
    g = B.newton_zero(0, 2.4)
    print(f"\nNewton depuis 2.4 : j_0,1 = {g:.12f}  (scipy : {z[0][0]:.12f})")
    print("Zéros de J'_1 (sommets des bosses) :", np.round(B.zeros_derivee(1, 3), 4))
    montrer("zeros")


def cas_spheriques(montrer):
    titre("9. ORDRES DEMI-ENTIERS : Bessel sphériques = sin et cos")
    para("""
j_n(x) = √(π/2x) · J_(n+1/2)(x) : pour ces ordres, plus de Γ ni de séries
infinies. j_0 = sin x / x, j_1 = sin x/x² − cos x/x, etc. Elles décrivent la
diffusion des ondes et l'atome d'hydrogène (partie radiale).
""")
    x = np.linspace(0.5, 30, 200)
    for n in range(4):
        print(f"   j_{n} : |fermée − J_(n+½)| = "
              f"{maxerr(B.sph_j_fermee(n, x), B.sph_j_depuis_J(n, x)):.1e},  |J_(n+½) − scipy| = "
              f"{maxerr(B.sph_j_depuis_J(n, x), special.spherical_jn(n, x)):.1e}")
    r = B.sph_j_recurrence(5, 9.0)
    print(f"\n   récurrence j_(n+1) = (2n+1)/x·j_n − j_(n−1) à x = 9 : "
          f"erreur {maxerr(r, special.spherical_jn(np.arange(6), 9.0)):.1e}")
    montrer("spheriques")


def cas_tambour(montrer):
    titre("10. LE TAMBOUR : pourquoi il n'a pas de note claire")
    para("""
Une peau circulaire de rayon 1 vibre selon u = J_m(α r) cos(mθ), avec
J_m(α) = 0 (bord fixe). Les fréquences sont proportionnelles aux zéros de J_m,
qui ne sont PAS des multiples entiers du fondamental : le son est inharmonique.
""")
    j01 = special.jn_zeros(0, 1)[0]
    print(f"{'mode (m,n)':>12}{'α':>10}{'fréquence / fondamental':>26}")
    modes = sorted(((special.jn_zeros(m, n)[-1], m, n) for m in range(4) for n in (1, 2)))
    for a, m, n in modes:
        print(f"{'(' + str(m) + ',' + str(n) + ')':>12}{a:10.4f}{a / j01:26.4f}")
    print("\nCorde vibrante, elle : 1, 2, 3, 4... (harmoniques entières).")
    print("Chaîne pesante suspendue (Bernoulli 1732, la 1re apparition de J_0) :",
          np.round(B.pendule_corde_pesante(k=4), 4))
    montrer("tambour")


def cas_fourier_bessel(montrer):
    titre("11. FOURIER–BESSEL : la chaleur dans un cylindre")
    para("""
Sur [0,1], les J_ν(α_j r) (α_j = j-ième zéro de J_ν) sont orthogonales avec le
poids r : on développe donc toute fonction comme une série de Fourier.
Application : un cylindre chaud dont le bord est maintenu à 0 refroidit selon
u(r,t) = Σ c_j J_0(α_j r) e^(−α_j² t).
""")
    f = lambda r: 1 - r ** 2
    alphas, c = B.coefficients_fourier_bessel(f, 0, 8)
    print("Coefficients c_j de 1 − r² :", np.round(c, 5))
    r = np.linspace(0, 1, 200)
    for k in (1, 2, 4, 8):
        approx = sum(cj * special.j0(a * r) for a, cj in zip(alphas[:k], c[:k]))
        print(f"   {k} terme(s) : erreur max {maxerr(approx, f(r)):.2e}")
    print("\nOrthogonalité : ∫₀¹ r J_0(α_i r) J_0(α_k r) dr :")
    from scipy import integrate
    for i, k in [(0, 1), (1, 2), (0, 0)]:
        v, _ = integrate.quad(lambda x: x * special.j0(alphas[i] * x) * special.j0(alphas[k] * x), 0, 1)
        print(f"   (i={i}, k={k}) : {v:.6f}" + (f"   (attendu J_1(α)²/2 = {special.j1(alphas[i]) ** 2 / 2:.6f})" if i == k else ""))
    print(f"\nTempérature au centre : t=0 → {B.chaleur_cylindre(alphas, c, 0.0, 0):.4f}, "
          f"t=0.1 → {B.chaleur_cylindre(alphas, c, 0.0, 0.1):.4f}, "
          f"t=0.3 → {B.chaleur_cylindre(alphas, c, 0.0, 0.3):.4f}")
    montrer("fourier_bessel")


def cas_diffraction(montrer):
    titre("12. DIFFRACTION : la tache d'Airy")
    para("""
Une étoile vue dans un télescope n'est jamais un point : c'est la figure
I(x) = (2 J_1(x)/x)², avec x = π D sinθ / λ. Le premier anneau sombre est le
premier zéro de J_1, ce qui donne le critère de Rayleigh θ = 1,22 λ/D.
""")
    z = special.jn_zeros(1, 3)
    print("Zéros de J_1 (anneaux sombres) :", np.round(z, 4))
    print(f"Premier zéro / π = {z[0] / np.pi:.4f}  → θ_min ≈ 1,22 λ/D")
    x = np.array([0.0, 3.8317, 5.1356, 7.0156, 8.4172])
    print("Intensité en x =", x, "→", np.round(B.airy(x), 5))
    frac = 1 - special.j0(z[0]) ** 2 - special.j1(z[0]) ** 2
    print(f"Fraction de lumière dans le disque central : {frac:.4f}  (≈ 84 %)")
    montrer("airy")


def cas_fm(montrer):
    titre("13. SYNTHÈSE SONORE FM : Bessel dans ton synthétiseur")
    para("""
cos(ω_p t + β sin ω_m t) = Σ J_n(β) cos((ω_p + nω_m) t) : moduler la fréquence
d'un son crée une infinité de raies latérales d'amplitude J_n(β). C'est le
principe du synthétiseur Yamaha DX7. On le vérifie avec une FFT.
""")
    fs, fp, fm, beta = 8192, 1000.0, 100.0, 2.0
    t = np.arange(fs) / fs
    sig = B.signal_fm(beta, fp, fm, t)
    spec = np.abs(np.fft.rfft(sig)) * 2 / fs
    print(f"β = {beta} ; porteuse 1000 Hz, modulante 100 Hz :")
    print(f"{'raie':>6}{'f (Hz)':>9}{'|FFT|':>10}{'|J_n(β)|':>11}")
    for n in range(0, 5):
        f = int(fp + n * fm)
        print(f"{n:6d}{f:9d}{spec[f]:10.5f}{abs(special.jv(n, beta)):11.5f}")
    z = special.jn_zeros(0, 1)[0]
    sig0 = B.signal_fm(z, fp, fm, t)
    print(f"\nβ = {z:.4f} (zéro de J_0) : amplitude à 1000 Hz = "
          f"{abs(np.fft.rfft(sig0))[int(fp)] * 2 / fs:.1e} → la porteuse a disparu !")
    montrer("fm")


def cas_kepler(montrer):
    titre("14. KEPLER : le problème qui a fait naître Bessel (1824)")
    para("""
Position d'une planète : E − e sin E = M (équation de Kepler). Bessel a montré
E = M + 2 Σ J_n(n e)/n · sin(nM). Le développement en puissances de e diverge
au-delà de 0,6627 (limite de Laplace) ; la série de Bessel converge pour tout
e < 1, mais de plus en plus lentement quand e approche de 1.
""")
    M = np.linspace(0, 2 * np.pi, 400)
    print(f"{'e':>6}{'1 terme':>12}{'3 termes':>12}{'10 termes':>12}{'40 termes':>12}")
    for e in (0.1, 0.3, 0.6, 0.9):
        ref = B.kepler_newton(M, e)
        errs = [maxerr(B.kepler_bessel(M, e, N), ref) for N in (1, 3, 10, 40)]
        print(f"{e:6.1f}" + "".join(f"{v:12.2e}" for v in errs))
    e, m = 0.0167, 1.0     # la Terre
    print(f"\nOrbite terrestre (e = {e}) : E = {float(B.kepler_bessel(np.array([m]), e)[0]):.8f} rad "
          f"(Newton : {float(B.kepler_newton(np.array([m]), e)[0]):.8f})")
    montrer("kepler")


def cas_peau(montrer):
    titre("15. EFFET DE PEAU : J_0 de variable complexe")
    para("""
Un courant alternatif dans un fil cylindrique vérifie l'équation de Bessel avec
k² = −iωμσ : J_0 évaluée en argument COMPLEXE. Plus la fréquence monte, plus le
courant se rue à la surface : le cœur du fil ne sert presque à rien.
""")
    print(f"{'ω μ σ R²':>10}{'|J(0)/J(R)|':>14}{'|J(R/2)/J(R)|':>16}")
    for w in (1, 30, 100, 300, 1000):
        print(f"{w:10d}{float(B.effet_de_peau(0.0, w)):14.4f}{float(B.effet_de_peau(0.5, w)):16.4f}")
    montrer("peau")


CAS = {
    "premieres": (cas_premieres, "J_n, Y_n, parité, singularité"),
    "equation": (cas_equation, "l'équation de Bessel, le Wronskien, ordres réels"),
    "serie": (cas_serie, "la série entière et sa perte de précision"),
    "integrale": (cas_integrale, "intégrale de Bessel, génératrice, Jacobi–Anger"),
    "recurrences": (cas_recurrences, "récurrences, algorithme de Miller"),
    "asymptotique": (cas_asymptotique, "grands x, approximation de Hankel"),
    "modifiees": (cas_modifiees, "I_n et K_n"),
    "zeros": (cas_zeros, "zéros, McMahon, Newton"),
    "spheriques": (cas_spheriques, "ordres demi-entiers, sin et cos"),
    "tambour": (cas_tambour, "modes d'un tambour, chaîne pesante"),
    "fourier_bessel": (cas_fourier_bessel, "orthogonalité, chaleur dans un cylindre"),
    "diffraction": (cas_diffraction, "tache d'Airy, critère de Rayleigh"),
    "fm": (cas_fm, "synthèse sonore FM"),
    "kepler": (cas_kepler, "l'équation de Kepler"),
    "peau": (cas_peau, "effet de peau, argument complexe"),
}

# nom du cas -> nom de la figure (fig_<nom> dans illustrations.py)
_FIGURE = {"diffraction": "airy"}


def lancer(nom, montrer):
    CAS[nom][0](lambda fig, **kw: montrer(_FIGURE.get(fig, fig), **kw))


def liste():
    print("Cas disponibles :")
    for nom, (_, desc) in CAS.items():
        print(f"   {nom:<15} {desc}")


def menu(montrer):
    noms = list(CAS)
    while True:
        print("\n=== Le laboratoire de Bessel ===")
        for i, nom in enumerate(noms, 1):
            print(f" {i:2d}. {nom:<15} {CAS[nom][1]}")
        print("  q. quiz     t. tout     0. quitter")
        try:
            choix = input("Ton choix : ").strip().lower()
        except EOFError:
            return
        if choix in ("0", ""):
            return
        if choix == "t":
            for n in noms:
                lancer(n, montrer)
        elif choix == "q":
            import quiz
            quiz.jouer(8)
        elif choix.isdigit() and 1 <= int(choix) <= len(noms):
            lancer(noms[int(choix) - 1], montrer)
        else:
            print("Choix invalide.")


def main(argv=None):
    p = argparse.ArgumentParser(description="Le laboratoire de Bessel.")
    p.add_argument("--cas", action="append", choices=list(CAS), help="cas à montrer (répétable)")
    p.add_argument("--tout", action="store_true", help="tous les cas")
    p.add_argument("--liste", action="store_true", help="liste les cas")
    p.add_argument("--sauver", metavar="DOSSIER", help="enregistre les figures en PNG")
    p.add_argument("--sans-graphique", action="store_true", help="texte seulement")
    p.add_argument("--quiz", type=int, metavar="N", help="quiz de N questions")
    p.add_argument("--graine", type=int, help="graine aléatoire du quiz")
    a = p.parse_args(argv)

    if a.liste:
        liste()
        return 0
    montrer = Afficheur(graphique=not a.sans_graphique, dossier=a.sauver)
    if a.quiz:
        import quiz
        quiz.jouer(a.quiz, a.graine)
    if a.tout or a.cas:
        for nom in (list(CAS) if a.tout else a.cas):
            lancer(nom, montrer)
    elif not a.quiz:
        menu(montrer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
