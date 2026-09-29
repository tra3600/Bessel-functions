"""Le Défi de Bessel : un quiz dont chaque réponse est *calculée*, pas recopiée."""

import random

import numpy as np
from scipy import special

import bessel as B


def _choix(rng, bonne, distracteurs):
    uniques = list(dict.fromkeys(d for d in distracteurs if d != bonne))
    options = [bonne] + uniques[:3]
    rng.shuffle(options)
    return options


def q_zero(rng):
    n, k = rng.choice([0, 1, 2, 3]), rng.choice([1, 2, 3])
    z = special.jn_zeros(n, k)[-1]
    bonne = f"{z:.3f}"
    autres = [f"{v:.3f}" for v in (z + 0.9, z - 0.7, z * 1.3, z + 1.7)]
    return (f"Quel est le zéro numéro {k} de J_{n} (arrondi à 10⁻³) ?",
            _choix(rng, bonne, autres), bonne,
            f"jn_zeros({n}, {k}) = {z:.6f}. Les zéros de J_n sont espacés d'environ π.")


def q_valeur(rng):
    n, x = rng.choice([0, 1, 2]), rng.choice([1.0, 2.0, 5.0, 10.0])
    v = special.jv(n, x)
    bonne = f"{v:+.3f}"
    autres = [f"{w:+.3f}" for w in (-v, v + 0.25, special.jv(n + 1, x), v * 2)]
    return (f"Que vaut J_{n}({x:g}) (à 10⁻³ près) ?",
            _choix(rng, bonne, autres), bonne,
            f"J_{n}({x:g}) = {v:.6f} (série ou Miller, peu importe : c'est calculé).")


def q_recurrence(rng):
    n, x = rng.choice([1, 2, 3, 4]), rng.choice([2.0, 3.0, 7.0])
    v = 2 * n / x * special.jv(n, x) - special.jv(n - 1, x)
    bonne = f"{v:+.3f}"
    autres = [f"{w:+.3f}" for w in (special.jv(n - 1, x), -v, special.jv(n, x), v + 0.3)]
    return (f"Sachant J_{n - 1}({x:g}) = {special.jv(n - 1, x):+.3f} et "
            f"J_{n}({x:g}) = {special.jv(n, x):+.3f}, la récurrence "
            f"J_(n+1) = (2n/x)·J_n − J_(n−1) donne J_{n + 1}({x:g}) ≈ ?",
            _choix(rng, bonne, autres), bonne,
            f"J_{n + 1}({x:g}) = {special.jv(n + 1, x):+.3f}. Attention : monter la "
            "récurrence devient instable quand n > x (d'où Miller).")


def q_parite(rng):
    n = rng.choice([1, 2, 3, 4, 5, 6])
    x = 2.5
    bonne = "J_n(x)" if n % 2 == 0 else "−J_n(x)"
    return (f"Que vaut J_n(−x) pour n = {n} ?",
            _choix(rng, bonne, ["J_n(x)", "−J_n(x)", "0", "1/J_n(x)"]), bonne,
            f"J_n(−x) = (−1)^n J_n(x) ; ici (−1)^{n} = {(-1) ** n:+d} "
            f"(vérif. : {special.jv(n, -x):+.4f} contre {special.jv(n, x):+.4f}).")


def q_demi_entier(rng):
    bonne = "sin(x)/x"
    return ("La fonction de Bessel sphérique j_0(x) = sqrt(π/2x)·J_(1/2)(x) est égale à :",
            _choix(rng, bonne, ["sin(x)/x", "cos(x)/x", "sin(x)", "e^(−x)/x"]), bonne,
            "Pour les ordres demi-entiers, Bessel se réduit à sin et cos : "
            f"en x=2, {B.sph_j_depuis_J(0, 2.0):.6f} = sin(2)/2 = {np.sin(2) / 2:.6f}.")


def q_airy(rng):
    z = special.jn_zeros(1, 1)[0]
    bonne = f"{z / np.pi:.2f}"
    return ("Une ouverture circulaire de diamètre D donne un premier anneau sombre à "
            "θ = c·λ/D. Que vaut c (critère de Rayleigh) ?",
            _choix(rng, bonne, ["0.61", "1.00", "1.22", "2.44"]), bonne,
            f"Premier zéro de J_1 : {z:.4f}, et {z:.4f}/π = {z / np.pi:.3f} ≈ 1,22.")


def q_comportement(rng):
    quest = [
        ("Quelle fonction est infinie en x = 0 ?", "Y_0", ["J_0", "Y_0", "I_0", "J_1"],
         "Y_n ~ ln x (n = 0) ou x^(−n) : c'est la solution singulière."),
        ("Laquelle croît exponentiellement quand x → ∞ ?", "I_0",
         ["J_0", "K_0", "I_0", "Y_0"],
         "I_0(x) ~ e^x / sqrt(2πx) ; J_0 et Y_0 oscillent, K_0 décroît."),
        ("Quelle enveloppe borne J_n(x) pour x grand ?", "sqrt(2/(πx))",
         ["1/x", "sqrt(2/(πx))", "e^(−x)", "1/ln x"],
         "J_n(x) ≈ sqrt(2/πx)·cos(x − nπ/2 − π/4) : l'amplitude décroît en 1/√x."),
        ("Que vaut la somme J_0(x)² + 2·Σ J_n(x)² pour tout x ?", "1",
         ["0", "1", "x", "J_0(2x)"],
         f"Conservation de l'énergie : en x=7.3 → {float(B.somme_carres(np.array(7.3))):.12f}."),
    ]
    enonce, bonne, opts, expl = rng.choice(quest)
    return enonce, _choix(rng, bonne, opts), bonne, expl


def q_fm(rng):
    z = special.jn_zeros(0, 1)[0]
    bonne = f"{z:.4f}"
    return ("En synthèse FM, pour quel indice β la fréquence porteuse disparaît-elle "
            "du spectre (amplitude J_0(β) = 0) ?",
            _choix(rng, bonne, ["1.0000", "2.4048", "3.1416", "3.8317"]), bonne,
            "Amplitude de la raie n : J_n(β). Le premier zéro de J_0 est 2,4048.")


def q_tambour(rng):
    z = special.jn_zeros(0, 1)[0]
    m, n = rng.choice([(1, 1), (2, 1), (0, 2)])
    ratio = special.jn_zeros(m, n)[-1] / z
    bonne = f"{ratio:.3f}"
    autres = [f"{w:.3f}" for w in (ratio + 0.4, ratio - 0.3, 2.0, 1.0)]
    return (f"Un tambour circulaire : le mode ({m},{n}) est plus aigu que le mode "
            f"fondamental (0,1) d'un facteur (j_{m},{n} / j_0,1) ≈ ?",
            _choix(rng, bonne, autres), bonne,
            "Les harmoniques d'un tambour ne sont PAS entières (d'où son timbre sans "
            "hauteur nette) : contrairement à une corde.")


QUESTIONS = [q_zero, q_valeur, q_recurrence, q_parite, q_demi_entier, q_airy,
             q_comportement, q_fm, q_tambour]


def generer(n, graine=None):
    """n questions (énoncé, choix, bonne réponse, explication), reproductibles."""
    rng = random.Random(graine)
    fabriques = [QUESTIONS[i % len(QUESTIONS)] for i in range(n)]
    rng.shuffle(fabriques)
    return [f(rng) for f in fabriques]


def jouer(n=8, graine=None, lire=input, ecrire=print):
    """Pose n questions ; renvoie le score. `lire`/`ecrire` sont injectables (tests)."""
    score = 0
    for i, (enonce, choix, bonne, expl) in enumerate(generer(n, graine), 1):
        ecrire(f"\nQuestion {i}/{n} : {enonce}")
        for lettre, c in zip("ABCD", choix):
            ecrire(f"   {lettre}) {c}")
        try:
            rep = lire("Ta réponse (A-D) : ").strip().upper()[:1]
        except EOFError:
            ecrire("\n(fin de l'entrée)")
            break
        idx = "ABCD".find(rep)
        if idx != -1 and idx < len(choix) and choix[idx] == bonne:
            score += 1
            ecrire(f"   ✅ Exact ! {expl}")
        else:
            ecrire(f"   ❌ C'était {bonne}. {expl}")
    ecrire(f"\nScore : {score}/{n}")
    return score
