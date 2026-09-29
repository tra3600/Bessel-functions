"""Les figures du laboratoire : chaque fonction retourne une figure matplotlib."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

import bessel as B

COULEURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _grille(ax):
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.6)


def fig_premieres(ordres=(0, 1, 2, 3), x_max=20):
    """Le cas historique du dépôt : J_n(x) et son cousin Y_n(x)."""
    x = np.linspace(0, x_max, 600)
    xs = np.linspace(0.05, x_max, 600)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    for n, c in zip(ordres, COULEURS):
        a.plot(x, special.jv(n, x), color=c, label=f"$J_{n}(x)$")
        b.plot(xs, special.yv(n, xs), color=c, label=f"$Y_{n}(x)$")
    a.plot(x[1:], B.enveloppe(x[1:]), "k:", lw=1, label=r"$\pm\sqrt{2/\pi x}$")
    a.plot(x[1:], -B.enveloppe(x[1:]), "k:", lw=1)
    a.set(title="Première espèce : régulière en 0", xlabel="x", ylabel="$J_n(x)$")
    b.set(title="Deuxième espèce : singulière en 0", xlabel="x", ylabel="$Y_n(x)$",
          ylim=(-1.5, 1))
    for ax in (a, b):
        _grille(ax)
        ax.legend()
    fig.tight_layout()
    return fig


def fig_serie(x_max=30):
    """La série converge toujours... mais la précision s'effondre pour x grand."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.linspace(0, 15, 500)
    a.plot(x, special.j0(x), "k", lw=2.5, label="$J_0$ exacte")
    for N, c in zip((2, 4, 8, 14), COULEURS):
        a.plot(x, B.serie_J(0, x, termes=N), color=c, label=f"{N} termes")
    a.set(ylim=(-1.5, 1.5), title="Sommes partielles de la série", xlabel="x")
    xx = np.linspace(1, x_max, 300)
    for N, c in zip((30, 60, 120), COULEURS):
        err = np.abs(B.serie_J(0, xx, termes=N) - special.j0(xx)) + 1e-18
        b.semilogy(xx, err, color=c, label=f"{N} termes")
    b.set(title="Erreur : annulations catastrophiques", xlabel="x", ylabel="|erreur|")
    for ax in (a, b):
        _grille(ax)
        ax.legend()
    fig.tight_layout()
    return fig


def fig_modifiees():
    """I_n croît comme e^x, K_n décroît comme e^{−x} : pas d'oscillation."""
    x = np.linspace(0.05, 6, 400)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    for n, c in zip((0, 1, 2, 3), COULEURS):
        a.semilogy(x, special.iv(n, x), color=c, label=f"$I_{n}$")
        b.semilogy(x, special.kv(n, x), color=c, label=f"$K_{n}$")
    a.semilogy(x, np.exp(x) / np.sqrt(2 * np.pi * x), "k:", label=r"$e^x/\sqrt{2\pi x}$")
    b.semilogy(x, np.sqrt(np.pi / (2 * x)) * np.exp(-x), "k:",
               label=r"$\sqrt{\pi/2x}\,e^{-x}$")
    a.set(title="$I_n$ : croissance exponentielle", xlabel="x", ylim=(1e-4, 500))
    b.set(title="$K_n$ : décroissance exponentielle", xlabel="x", ylim=(1e-4, 100))
    for ax in (a, b):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig


def fig_asymptotique():
    """Approximation de Hankel : quelques termes suffisent dès x ≈ 3."""
    x = np.linspace(1, 20, 500)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    a.plot(x, special.j0(x), "k", lw=2.5, label="$J_0$ exacte")
    for t, c in zip((1, 2, 4), COULEURS):
        a.plot(x, B.asymptotique_J(0, x, termes=t), "--", color=c,
               label=f"Hankel, {t} terme(s)")
    a.set(ylim=(-1, 1.1), title="Approximation de Hankel", xlabel="x")
    for n, c in zip((0, 1, 5), COULEURS):
        err = np.abs(B.asymptotique_J(n, x, termes=4) - special.jv(n, x)) + 1e-18
        b.semilogy(x, err, color=c, label=f"n = {n}")
    b.set(title="Erreur (4 termes)", xlabel="x")
    for ax in (a, b):
        _grille(ax)
        ax.legend()
    fig.tight_layout()
    return fig


def fig_recurrences(n_max=40, x=10.0):
    """Récurrence montante (instable) contre descendante de Miller (stable)."""
    n = np.arange(n_max + 1)
    exact = special.jv(n, x)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(n, np.abs(B.recurrence_montante(x, n_max) - exact) + 1e-20, "o-",
                label="montante (instable)")
    ax.semilogy(n, np.abs(B.miller(x, n_max) - exact) + 1e-20, "s-",
                label="Miller descendante (stable)")
    ax.axvline(x, color="k", ls=":")
    ax.set(title=f"Erreur sur $J_n({x:g})$ : l'instabilité démarre pour n > x",
           xlabel="n", ylabel="|erreur|")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_zeros():
    """Zéros de J_n et de J_n' ; la méthode de McMahon."""
    x = np.linspace(0, 25, 800)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    for n, c in zip((0, 1, 2), COULEURS):
        a.plot(x, special.jv(n, x), color=c, label=f"$J_{n}$")
        z = B.zeros_J(n, 7)
        a.plot(z, np.zeros_like(z), "o", color=c)
    a.set(title="Zéros : entrelacés (les zéros de $J_1$ séparent ceux de $J_0$)",
          xlabel="x")
    k = np.arange(1, 11)
    for n, c in zip((0, 1, 2), COULEURS):
        err = np.abs(B.mcmahon(n, k) - B.zeros_J(n, 10)) + 1e-18
        b.semilogy(k, err, "o-", color=c, label=f"n = {n}")
    b.set(title="Erreur de l'approximation de McMahon", xlabel="k-ième zéro")
    for ax in (a, b):
        _grille(ax)
        ax.legend()
    fig.tight_layout()
    return fig


def fig_spheriques():
    """j_n(x) : des sinus et des cosinus déguisés, pour la diffusion et l'atome."""
    x = np.linspace(0.01, 20, 600)
    fig, ax = plt.subplots(figsize=(9, 5))
    for n, c in zip(range(4), COULEURS):
        ax.plot(x, B.sph_j_fermee(n, x), color=c, label=f"$j_{n}(x)$")
        ax.plot(x, B.sph_j_depuis_J(n, x), "k:", lw=1)
    ax.set(title=r"Sphériques (pointillés : $\sqrt{\pi/2x}\,J_{n+1/2}$)", xlabel="x")
    _grille(ax)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_tambour(modes=((0, 1), (1, 1), (2, 1), (0, 2), (3, 1), (1, 2))):
    """Les vibrations d'une peau de tambour circulaire."""
    r = np.linspace(0, 1, 120)
    th = np.linspace(0, 2 * np.pi, 240)
    R, T = np.meshgrid(r, th)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), subplot_kw={"projection": "polar"})
    for ax, (m, n) in zip(axes.ravel(), modes):
        Z, alpha = B.mode_tambour(m, n, R, T)
        ax.pcolormesh(T, R, Z, cmap="RdBu", shading="auto",
                      vmin=-np.max(abs(Z)), vmax=np.max(abs(Z)))
        ax.set_title(f"mode ({m},{n}) : ×{alpha / special.jn_zeros(0, 1)[0]:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Modes propres d'un tambour (rouge/bleu : la peau monte/descend)")
    fig.tight_layout()
    return fig


def fig_fourier_bessel(k_max=6):
    """Développer 1 − r² sur la base J_0(α_j r), puis la voir refroidir."""
    f = lambda r: 1 - r ** 2
    alphas, c = B.coefficients_fourier_bessel(f, 0, k_max)
    r = np.linspace(0, 1, 300)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    a.plot(r, f(r), "k", lw=2.5, label="$1-r^2$")
    for k in (1, 2, 4, k_max):
        approx = sum(cj * special.j0(al * r) for al, cj in zip(alphas[:k], c[:k]))
        a.plot(r, approx, "--", label=f"{k} terme(s)")
    a.set(title="Série de Fourier–Bessel", xlabel="r")
    for t in (0, 0.02, 0.05, 0.1, 0.2):
        b.plot(r, B.chaleur_cylindre(alphas, c, r, t), label=f"t = {t}")
    b.set(title="Refroidissement d'un cylindre", xlabel="r", ylabel="température")
    for ax in (a, b):
        _grille(ax)
        ax.legend()
    fig.tight_layout()
    return fig


def fig_airy():
    """La tache d'Airy : pourquoi un télescope ne voit pas les étoiles ponctuelles."""
    x = np.linspace(0, 15, 600)
    lim = 10
    g = np.linspace(-lim, lim, 500)
    X, Y = np.meshgrid(g, g)
    I = B.airy(np.hypot(X, Y))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    a.imshow(I ** 0.3, extent=(-lim, lim, -lim, lim), cmap="magma")
    a.set(title="Tache d'Airy (contraste renforcé)", xlabel="x", ylabel="y")
    b.semilogy(x, B.airy(x) + 1e-12)
    for z in B.zeros_J(1, 3):
        b.axvline(z, color="r", ls=":")
    b.set(title=r"$(2J_1(x)/x)^2$ ; premier zéro 3,8317 → 1,22 λ/D",
          xlabel="x", ylim=(1e-6, 1.5))
    b.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_fm(betas=(0.5, 2.4048, 5.0)):
    """Synthèse FM : l'indice de modulation β décide des raies J_n(β)."""
    fig, axes = plt.subplots(1, len(betas), figsize=(12, 4), sharey=True)
    for ax, beta in zip(axes, betas):
        n, amp = B.bandes_laterales_fm(beta, 8)
        ax.stem(n, amp, basefmt=" ")
        ax.set(title=f"β = {beta:g}", xlabel="raie  f_p + n·f_m")
        _grille(ax)
    axes[0].set_ylabel("amplitude $J_n(β)$")
    fig.suptitle("β = 2,4048 (zéro de $J_0$) : la porteuse disparaît !")
    fig.tight_layout()
    return fig


def fig_kepler(e=0.6):
    """Kepler : la série de Bessel de 1824 contre la méthode de Newton."""
    M = np.linspace(0, 2 * np.pi, 300)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    ref = B.kepler_newton(M, e)
    for N, c in zip((1, 3, 10), COULEURS):
        a.plot(M, B.kepler_bessel(M, e, N), color=c, label=f"{N} terme(s)")
        b.semilogy(M, np.abs(B.kepler_bessel(M, e, N) - ref) + 1e-18, color=c,
                   label=f"{N} terme(s)")
    a.plot(M, ref, "k:", label="Newton")
    a.set(title=f"Anomalie excentrique, e = {e}", xlabel="M", ylabel="E")
    b.set(title="Erreur de la série de Bessel", xlabel="M")
    for ax in (a, b):
        _grille(ax)
        ax.legend()
    fig.tight_layout()
    return fig


def fig_peau():
    """Effet de peau : le courant fuit le cœur du fil quand la fréquence monte."""
    r = np.linspace(0, 1, 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    for w in (1, 30, 100, 300, 1000):
        ax.plot(r, B.effet_de_peau(r, w), label=f"ω μ σ R² = {w}")
    ax.set(title="Densité de courant dans un fil cylindrique", xlabel="r / R",
           ylabel="|J(r)/J(R)|")
    _grille(ax)
    ax.legend()
    fig.tight_layout()
    return fig
