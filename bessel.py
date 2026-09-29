"""Le cœur mathématique : les fonctions de Bessel, calculées « à la main ».

Chaque fonction ici est une implémentation indépendante (série, intégrale,
récurrence, asymptotique...) que les cas illustrés et les tests comparent à
`scipy.special`, la référence.

Rappel : J_nu et Y_nu résolvent l'équation de Bessel
        x² y'' + x y' + (x² − nu²) y = 0,
et I_nu, K_nu l'équation « modifiée »  x² y'' + x y' − (x² + nu²) y = 0.
"""

import numpy as np
from scipy import integrate, special

# ----------------------------------------------------------------------
# 1. La série entière (valable pour tout ordre réel, mais instable si x grand)
# ----------------------------------------------------------------------
def _serie(nu, x, signe, termes):
    x = np.asarray(x, dtype=float)
    demi = x / 2.0
    terme = demi ** nu / special.gamma(nu + 1.0)
    somme = terme.copy()
    for k in range(1, termes):
        terme = signe * terme * demi ** 2 / (k * (k + nu))
        somme = somme + terme
    return somme


def termes_necessaires(x_max):
    """Nombre de termes de la série pour atteindre la précision machine."""
    return int(2 * float(x_max) + 30)


def serie_J(nu, x, termes=None):
    """J_nu(x) = Σ (−1)^k / (k! Γ(k+nu+1)) (x/2)^(2k+nu)."""
    x = np.asarray(x, dtype=float)
    termes = termes or termes_necessaires(np.max(np.abs(x)))
    return _serie(nu, x, -1.0, termes)


def serie_I(nu, x, termes=None):
    """I_nu(x) = Σ 1 / (k! Γ(k+nu+1)) (x/2)^(2k+nu)  (tous les signes +)."""
    x = np.asarray(x, dtype=float)
    termes = termes or termes_necessaires(np.max(np.abs(x)))
    return _serie(nu, x, 1.0, termes)


# ----------------------------------------------------------------------
# 2. L'intégrale de Bessel (ordre entier)
# ----------------------------------------------------------------------
def integrale_J(n, x, points=256):
    """J_n(x) = 1/(2π) ∫₀^{2π} cos(nτ − x sin τ) dτ  (méthode des trapèzes).

    L'intégrande est périodique : les trapèzes convergent alors de façon
    exponentielle, 256 points suffisent jusqu'à x ≈ 100.
    """
    x = np.asarray(x, dtype=float)
    tau = np.linspace(0.0, 2 * np.pi, points, endpoint=False)
    xx = x[..., None]
    return np.mean(np.cos(n * tau - xx * np.sin(tau)), axis=-1)


# ----------------------------------------------------------------------
# 3. La récurrence descendante de Miller (ordre entier, x réel)
# ----------------------------------------------------------------------
def miller(x, n_max, marge=None):
    """[J_0(x), ..., J_n_max(x)] par récurrence *descendante*.

    J_{k-1} = (2k/x) J_k − J_{k+1} est instable quand on la monte (k > x) mais
    stable quand on la descend. On part de valeurs arbitraires (0, 1) à un
    rang M très élevé, puis on normalise avec  1 = J_0 + 2 Σ J_{2k}.
    """
    x = float(x)
    if x == 0.0:
        return np.array([1.0] + [0.0] * n_max)
    marge = marge or 40
    M = 2 * ((max(n_max, int(abs(x))) + marge + 1) // 2 + 1)   # M pair
    suivant, courant = 0.0, 1e-30
    valeurs = np.zeros(M + 1)
    valeurs[M] = courant
    for k in range(M, 0, -1):
        precedent = 2.0 * k / x * courant - suivant
        suivant, courant = courant, precedent
        valeurs[k - 1] = courant
        if abs(courant) > 1e250:                   # évite le débordement
            valeurs *= 1e-250
            courant, suivant = courant * 1e-250, suivant * 1e-250
    norme = valeurs[0] + 2.0 * np.sum(valeurs[2::2])
    return (valeurs / norme)[: n_max + 1]


def recurrence_montante(x, n_max):
    """La *mauvaise* idée : monter à partir de J_0 et J_1 (instable si n > x)."""
    x = float(x)
    v = [special.j0(x), special.j1(x)]
    for k in range(1, n_max):
        v.append(2.0 * k / x * v[k] - v[k - 1])
    return np.array(v[: n_max + 1])


# ----------------------------------------------------------------------
# 4. Relations exactes (résidus qui doivent valoir ~0)
# ----------------------------------------------------------------------
def residu_equation(nu, x):
    """Résidu de x²y'' + xy' + (x²−nu²)y pour y = J_nu, via les récurrences."""
    x = np.asarray(x, dtype=float)
    J = special.jv
    y = J(nu, x)
    yp = (J(nu - 1, x) - J(nu + 1, x)) / 2
    ypp = (J(nu - 2, x) - 2 * J(nu, x) + J(nu + 2, x)) / 4
    return x ** 2 * ypp + x * yp + (x ** 2 - nu ** 2) * y


def residu_equation_modifiee(nu, x):
    """Idem pour I_nu : x²y'' + xy' − (x²+nu²)y."""
    x = np.asarray(x, dtype=float)
    I = special.iv
    y = I(nu, x)
    yp = (I(nu - 1, x) + I(nu + 1, x)) / 2
    ypp = (I(nu - 2, x) + 2 * I(nu, x) + I(nu + 2, x)) / 4
    return x ** 2 * ypp + x * yp - (x ** 2 + nu ** 2) * y


def residu_recurrence(nu, x):
    """J_{nu−1} + J_{nu+1} − (2nu/x) J_nu  (doit être nul)."""
    x = np.asarray(x, dtype=float)
    return special.jv(nu - 1, x) + special.jv(nu + 1, x) - 2 * nu / x * special.jv(nu, x)


def residu_derivee(nu, x):
    """J'_nu − (J_{nu−1} − (nu/x) J_nu)  (doit être nul, dérivée calculée par scipy)."""
    x = np.asarray(x, dtype=float)
    return special.jvp(nu, x) - (special.jv(nu - 1, x) - nu / x * special.jv(nu, x))


def wronskien(nu, x):
    """W[J_nu, Y_nu] = J_{nu+1} Y_nu − J_nu Y_{nu+1}, qui vaut exactement 2/(πx)."""
    x = np.asarray(x, dtype=float)
    return special.jv(nu + 1, x) * special.yv(nu, x) - special.jv(nu, x) * special.yv(nu + 1, x)


def somme_carres(x, n_max=60):
    """J_0² + 2 Σ_{n≥1} J_n² = 1 pour tout x  (conservation de l'énergie)."""
    x = np.asarray(x, dtype=float)
    n = np.arange(1, n_max + 1)
    return special.jv(0, x) ** 2 + 2 * np.sum(special.jv(n, x[..., None]) ** 2, axis=-1)


def fonction_generatrice(x, t, n_max=60):
    """Σ_{n=−N}^{N} J_n(x) t^n  ≈  exp( x/2 (t − 1/t) )."""
    n = np.arange(-n_max, n_max + 1)
    return np.sum(special.jv(n, x) * np.power(t, n.astype(float)))


def jacobi_anger(x, theta, n_max=60):
    """Σ J_n(x) e^{inθ}  ≈  e^{i x sin θ}  (le lien avec les séries de Fourier)."""
    n = np.arange(-n_max, n_max + 1)
    theta = np.asarray(theta, dtype=float)
    return np.sum(special.jv(n, x) * np.exp(1j * n * theta[..., None]), axis=-1)


# ----------------------------------------------------------------------
# 5. Grands x : le développement asymptotique de Hankel
# ----------------------------------------------------------------------
def asymptotique_J(nu, x, termes=6):
    """J_nu(x) ≈ sqrt(2/πx) [P cos χ − Q sin χ], χ = x − (nu/2 + 1/4)π."""
    x = np.asarray(x, dtype=float)
    mu = 4.0 * nu ** 2
    a = [1.0]
    for k in range(1, 2 * termes + 2):
        a.append(a[-1] * (mu - (2 * k - 1) ** 2) / (k * 8.0))
    P = sum((-1) ** m * a[2 * m] / x ** (2 * m) for m in range(termes))
    Q = sum((-1) ** m * a[2 * m + 1] / x ** (2 * m + 1) for m in range(termes))
    chi = x - (nu / 2 + 0.25) * np.pi
    return np.sqrt(2 / (np.pi * x)) * (P * np.cos(chi) - Q * np.sin(chi))


def asymptotique_I(nu, x):
    """I_nu(x) ≈ e^x / sqrt(2πx) · (1 − (4nu²−1)/(8x))  pour x grand."""
    x = np.asarray(x, dtype=float)
    return np.exp(x) / np.sqrt(2 * np.pi * x) * (1 - (4 * nu ** 2 - 1) / (8 * x))


def asymptotique_K(nu, x):
    """K_nu(x) ≈ sqrt(π/2x) e^{−x} (1 + (4nu²−1)/(8x))  pour x grand."""
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.pi / (2 * x)) * np.exp(-x) * (1 + (4 * nu ** 2 - 1) / (8 * x))


def enveloppe(x):
    """L'enveloppe universelle sqrt(2/(πx)) : J_nu(x) oscille dans ±enveloppe."""
    return np.sqrt(2 / (np.pi * np.asarray(x, dtype=float)))


# ----------------------------------------------------------------------
# 6. Les zéros
# ----------------------------------------------------------------------
def zeros_J(nu, k):
    """Les k premiers zéros positifs de J_nu (ordre entier), par scipy."""
    return special.jn_zeros(int(nu), k)


def mcmahon(nu, k):
    """Approximation de McMahon du k-ième zéro (k = 1, 2, ...) de J_nu."""
    k = np.asarray(k, dtype=float)
    mu = 4.0 * nu ** 2
    beta = (k + nu / 2 - 0.25) * np.pi
    return (beta - (mu - 1) / (8 * beta)
            - 4 * (mu - 1) * (7 * mu - 31) / (3 * (8 * beta) ** 3))


def newton_zero(nu, x0, iterations=8):
    """Raffine un zéro de J_nu par Newton, avec J' = J_{nu−1} − (nu/x) J_nu."""
    x = float(x0)
    for _ in range(iterations):
        f = special.jv(nu, x)
        fp = special.jv(nu - 1, x) - nu / x * f
        x -= f / fp
    return x


def zeros_derivee(nu, k):
    """Zéros de J'_nu (les crêtes de J_nu) ; utile pour les modes TE des guides."""
    return special.jnp_zeros(int(nu), k)


# ----------------------------------------------------------------------
# 7. Fonctions de Bessel sphériques
# ----------------------------------------------------------------------
def sph_j_fermee(n, x):
    """Formes fermées de j_n (n = 0, 1, 2, 3) : rien que des sin et des cos !"""
    x = np.asarray(x, dtype=float)
    s, c = np.sin(x), np.cos(x)
    if n == 0:
        return s / x
    if n == 1:
        return s / x ** 2 - c / x
    if n == 2:
        return (3 / x ** 2 - 1) * s / x - 3 * c / x ** 2
    if n == 3:
        return (15 / x ** 3 - 6 / x) * s / x - (15 / x ** 2 - 1) * c / x
    raise ValueError("formes fermées disponibles pour n = 0..3")


def sph_j_depuis_J(n, x):
    """j_n(x) = sqrt(π / 2x) J_{n+1/2}(x)  : le lien avec les ordres demi-entiers."""
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.pi / (2 * x)) * special.jv(n + 0.5, x)


def sph_j_recurrence(n_max, x):
    """j_0..j_n_max par j_{n+1} = (2n+1)/x j_n − j_{n−1} (montante, x > n_max)."""
    x = float(x)
    j = [np.sin(x) / x, np.sin(x) / x ** 2 - np.cos(x) / x]
    for n in range(1, n_max):
        j.append((2 * n + 1) / x * j[n] - j[n - 1])
    return np.array(j[: n_max + 1])


# ----------------------------------------------------------------------
# 8. Applications
# ----------------------------------------------------------------------
def mode_tambour(m, n, r, theta):
    """Mode (m, n) d'un tambour circulaire de rayon 1 : J_m(α_mn r) cos(mθ).

    Retourne (déformation, α_mn) avec α_mn = n-ième zéro de J_m.
    """
    alpha = special.jn_zeros(m, n)[-1]
    return special.jv(m, alpha * r) * np.cos(m * theta), alpha


def coefficients_fourier_bessel(f, nu, k, quad_limit=200):
    """c_j = 2 / J_{nu+1}(α_j)² · ∫₀¹ x f(x) J_nu(α_j x) dx,  j = 1..k.

    Sur [0,1], les J_nu(α_j x) forment une base orthogonale (poids x) :
    f(x) = Σ c_j J_nu(α_j x).
    """
    alphas = special.jn_zeros(nu, k)
    c = []
    for a in alphas:
        val, _ = integrate.quad(lambda x: x * f(x) * special.jv(nu, a * x), 0, 1,
                                limit=quad_limit)
        c.append(2.0 * val / special.jv(nu + 1, a) ** 2)
    return alphas, np.array(c)


def chaleur_cylindre(alphas, c, r, t):
    """Température d'un cylindre refroidi à 0 sur le bord : Σ c_j J_0(α_j r) e^{−α_j² t}."""
    r = np.asarray(r, dtype=float)
    return sum(cj * special.j0(a * r) * np.exp(-a ** 2 * t) for a, cj in zip(alphas, c))


def airy(x):
    """Intensité de la tache d'Airy (ouverture circulaire) : (2 J_1(x)/x)²."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = (2 * special.j1(x[nz]) / x[nz]) ** 2
    return out


def bandes_laterales_fm(beta, n_max=8):
    """Amplitudes J_n(β) des raies aux fréquences f_p + n f_m (synthèse FM)."""
    n = np.arange(-n_max, n_max + 1)
    return n, special.jv(n, beta)


def signal_fm(beta, fp, fm, t):
    """cos(2π fp t + β sin 2π fm t)."""
    return np.cos(2 * np.pi * fp * t + beta * np.sin(2 * np.pi * fm * t))


def kepler_bessel(M, e, n_max=40):
    """Anomalie excentrique E (E − e sin E = M) par la série de Bessel (1824) :

        E = M + 2 Σ_{n≥1} J_n(n e)/n · sin(n M).

    C'est le problème pour lequel Bessel a introduit ces fonctions.
    """
    M = np.asarray(M, dtype=float)
    n = np.arange(1, n_max + 1)
    return M + 2 * np.sum(special.jv(n, n * e) / n * np.sin(np.outer(M, n)), axis=-1)


def kepler_newton(M, e, iterations=50):
    """Référence : l'équation de Kepler résolue par Newton."""
    M = np.asarray(M, dtype=float)
    E = M.copy() + e * np.sin(M)
    for _ in range(iterations):
        E -= (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def effet_de_peau(r, omega_mu_sigma):
    """Courant dans un fil : |J_0(k r)/J_0(k R)| avec k = sqrt(−i ω μ σ), R = 1.

    Aux hautes fréquences, le courant se concentre à la surface (effet de peau) ;
    ici J_0 est évaluée en argument complexe.
    """
    k = np.sqrt(-1j * omega_mu_sigma)
    r = np.asarray(r, dtype=float)
    return np.abs(special.jv(0, k * r) / special.jv(0, k))


def pendule_corde_pesante(L=1.0, k=3):
    """Fréquences propres d'une chaîne pesante suspendue (Bernoulli, 1732).

    Les modes sont y(s) = J_0(2ω sqrt(s/g)), avec  2ω sqrt(L/g) = j_{0,k}.
    Retourne les rapports ω_k/ω_1 = j_{0,k}/j_{0,1}, qui ne sont PAS des entiers.
    """
    z = special.jn_zeros(0, k)
    return z / z[0]
