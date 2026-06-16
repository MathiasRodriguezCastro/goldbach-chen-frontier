# proyectoGoldbach — La frontera Goldbach–Chen

Estudio **computacional y de optimización** de la transición entre las
representaciones de Goldbach ($N=p+q$, ambos primos) y las de Chen ($N=p+q$ con
$q$ primo **o** semiprimo). Implementa el programa propuesto en
[`paper/goldbach_chen_frontier.tex`](paper/goldbach_chen_frontier.tex) y lo
extiende con un hallazgo nuevo sobre la serie singular.

No es un intento de *probar* Goldbach: es evidencia reproducible, diagnósticos y
preguntas precisas en la frontera entre teoría analítica de números,
experimentación y optimización discreta.

## Resultado principal (nuevo)

Sea $\mathfrak S(N)=\prod_{p\mid N,p>2}\frac{p-1}{p-2}$ la serie singular de
Goldbach. Empíricamente, hasta $X=8\cdot10^6$:

| conteo | $\propto \mathfrak S(N)^{\beta}$ | $\beta$ medido |
|--------|----------------------------------|----------------|
| $R_1$ (primo+primo) | $\beta_1$ | **1.00** (Hardy–Littlewood, control) |
| $R_2$ (primo+semiprimo) | $\beta_2$ | **≈ 0.50** (raíz cuadrada) |

> **$R_2$ hereda solo la *raíz* de la serie singular de Goldbach.** La serie
> singular explica el ~92 % de la varianza de $\log(R_2/R_1)$. Consecuencia: la
> cuota relativa del canal semiprimo escala como $\mathfrak S(N)^{-1/2}$ — los
> semiprimos son proporcionalmente más abundantes justo donde Goldbach es
> *frágil*. Es una forma cuantitativa del **“Chen rescue effect”**.

Normalizar $R_2$ por $2C_2\,\mathfrak S(N)^{1/2}\,N\,W(N)/\log^2N$ (con $W$ una
suma de Mertens sobre el factor primo pequeño) reduce el coeficiente de variación
de **0.18** (heurística del `.tex`) a **0.018** — un ajuste 10× más fino.

Detalles, tablas, derivación y preguntas abiertas en
[`notes/theory.md`](notes/theory.md).

## Estructura

```
proyectoGoldbach/
├── paper/goldbach_chen_frontier.tex   ← propuesta original (el .tex)
├── src/
│   ├── sieve.py        ← cribas: primos, Omega, spf, semiprimos
│   ├── counts.py       ← R1, R2 por convolución FFT (validados vs fuerza bruta)
│   ├── heuristics.py   ← serie singular, peso de Mertens W(N), exponente de R2
│   ├── segmented.py    ← conteos por bloques a escala 10^9 (sin FFT global)
│   ├── layers.py       ← capas Ω(q)=k, R_cc, formas lineales N=m1·p+q
│   ├── balance.py      ← exponente de balance residual a_Res(N), r_star(N)
│   ├── transport.py    ← defecto de reflexión / transporte óptimo (Wasserstein)
│   ├── tda.py          ← persistencia 0-dim de valles del cometa
│   ├── dynamics.py     ← descomposición de varianza de theta(N)
│   ├── energy.py       ← energía de estado fundamental Ω(a)+Ω(b), termodinámica (Z, U, C, β*)
│   ├── diagnostics.py  ← theta, C(N), fragilidad, B(q), rescate L1/L2
│   ├── mip.py          ← MIP de selección (Gurobi): variación / rescate
│   └── plots.py        ← figuras
├── experiments/
│   ├── run_counts.py   ← driver: cribas→conteos→diagnósticos→figuras→summary.json
│   ├── run_mip.py      ← umbral de rescate + rama de mínima variación
│   ├── run_continuity.py ← continuidad débil de q/N
│   ├── run_scaling.py  ← β_2(X) hasta 10^9 (criba segmentada)
│   ├── run_layers.py   ← capas Ω(q)=k y exponente β_k
│   ├── run_balance.py  ← exponente de balance residual
│   ├── run_betalaw.py  ← validación de la ley β_2=1-c/⟨R2/R1⟩
│   ├── run_firstmoment.py← reducción rigurosa: respuestas A_ℓ,B_ℓ (Lema 3) + β_2^(1)
│   ├── run_exceptional.py← cota de 2º momento: concentración de R2/R1 en W~ (conjunto excepcional)
│   ├── run_transport.py← transporte óptimo / colapso de reflexión
│   ├── run_tda.py      ← persistencia de valles (Goldbach vs Chen)
│   ├── run_dynamics.py ← dinámica estadística de θ(N)
│   └── run_energy.py   ← energía de estado fundamental aritmética + termodinámica
├── notes/theory.md     ← hallazgos teóricos y experimentales
├── data/   (arrays .npz y summary.json — generados)
└── figures/(PNG — generados)
```

## Uso

```bash
# núcleo: conteos, diagnósticos y figuras (≈15 s para X=2e6)
python3 experiments/run_counts.py 2000000

# experimentos de optimización (rescate de Chen + rama MIP)
python3 experiments/run_mip.py

# continuidad débil de las medidas de q/N
python3 experiments/run_continuity.py

# tests rápidos de cada módulo
python3 src/sieve.py && python3 src/counts.py && python3 src/heuristics.py
```

Requisitos: `numpy`, `scipy`, `matplotlib`, `gurobipy` (licencia; solo para
`mip.py`/`run_mip.py`). El resto funciona sin Gurobi.

## Figuras generadas

| archivo | contenido |
|---------|-----------|
| `01_comets.png` | cometas de $R_1$, $R_2$ y excedente $C(N)$ |
| `02_singular_exponent.png` | **figura clave**: $R_1\propto\mathfrak S^1$ vs $R_2\propto\mathfrak S^{1/2}$ |
| `03_normalizacion.png` | colapso de las bandas de $A_1$ al dividir por $\mathfrak S(N)$ |
| `04_theta_C_W.png` | cuota $\theta(N)$ y $C(N)$ vs peso de Mertens $W(N)$ |
| `05_balance.png` | geometría de factores $B(q)$ de los semiprimos de Chen |
| `06_rescate.png` | umbral de rescate de Chen bajo restricción de balance |
| `07_rama_mip.png` | rama de mínima variación: suave en $s_N$, alterna de tipo |
| `08_continuidad.png` | continuidad débil de $q/N$: $\mu_N$ se estabiliza al suavizar; primo≈semiprimo≈uniforme |
| `09_beta_scaling.png` | **escalamiento a $10^9$**: $\beta_2(X)\approx-0.094+0.234\log\log X$ (el ½ es solo local) |
| `10_capas.png` | capas $\Omega(q)=k$: perfil $\rho(k)$ y exponente singular $\beta_k$ (cambia de signo) |
| `11_balance.png` | exponente de balance residual $a_{\rm Res}(N)$, $r_\star(N)$, correlación con $\mathfrak S(N)$ |
| `12_betalaw.png` | ley derivada $\beta_2=1-c/\langle R_2/R_1\rangle\to1$ (déficit $1/\log\log N$) |
| `13_transporte.png` | colapso de transporte: semiprimos reflejados matchean mejor a los primos (−22%) |
| `14_valles.png` | persistencia de valles: Chen NO rellena los valles de Goldbach (anomalías corr +0.64) |
| `15_dinamica.png` | $\theta(N)$ 94% determinista en $\mathfrak S(N)$; residuo no-blanco |

## Estado

- [x] Núcleo (cribas + conteos FFT) validado vs fuerza bruta; Goldbach verificado.
- [x] Serie singular: $\beta_1=1$ (control HL), $\beta_2\approx1/2$ (nuevo).
- [x] Diagnósticos: $\theta$, $C$, fragilidad, $B(q)$, rescate.
- [x] MIP: rama de mínima variación + umbral de rescate de Chen.
- [x] Continuidad débil de $\mu_N^{(t)}$: se estabiliza al suavizar; $q/N$ ≈ uniforme.
- [x] **Escalamiento a $10^9$** (criba segmentada): $\beta_2\approx-0.094+0.234\log\log X$.
- [x] Capas $\Omega(q)=k$: $\beta_k$ cambia de signo (realce $k\le2$, supresión $k\ge3$).
- [x] Exponente de balance residual $a_{\rm Res}(N)$: colapso típico + obstrucción local.
- [x] **Ley derivada $\beta_2=1-c/\langle R_2/R_1\rangle\to1$** (déficit $1/\log\log N$), validada al 1%.
- [x] Transporte óptimo (colapso de reflexión −22%), TDA de valles (Chen no rellena), dinámica de $\theta$ (94% determinista).
- [x] **Reducción rigurosa** (apéndice §A.5): $\beta_2\to1$ reducido de "HL puntual uniforme en $r$" a **Bombieri–Vinogradov + cota de conjunto excepcional para $R_2$** (à la Montgomery–Vaughan). Respuestas $A_\ell,B_\ell$ exactas (Lema 3).
- [x] **Teorema casi-todo-$N$** (apéndice §A.6): $\beta_2(N)\to1$ para casi todo $N$, **incondicional en outline** (la varianza usa solo el arco-menor de primos de Vinogradov + $\int|S_{\mathcal S_2}|^2=\#\mathcal S_2$; sin criba). Conjunto excepcional vacío al 5%.
- [ ] Column generation del MIP infinito (pricing real).
