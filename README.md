# proyectoGoldbach — La frontera Goldbach–Chen

Estudio **computacional y de optimización** de la transición entre las
representaciones de Goldbach ($N=p+q$, ambos primos) y las de Chen ($N=p+q$ con
$q$ primo **o** semiprimo). Implementa el programa propuesto en
[`paper/goldbach_chen_frontier.tex`](paper/goldbach_chen_frontier.tex) y lo
extiende con un hallazgo nuevo sobre la serie singular.

No es un intento de *probar* Goldbach: es evidencia reproducible, diagnósticos y
preguntas precisas en la frontera entre teoría analítica de números,
experimentación y optimización discreta.

## Resultado central

Sea $R_1(N)$ el nº de representaciones $N=p+q$ (ambos primos), $R_2(N)$ con $q$
semiprimo, y $\mathfrak S(N)=\prod_{p\mid N,p>2}\frac{p-1}{p-2}$ la serie singular de
Goldbach.

- $R_1$: $\mathfrak S(N)$ entra con exponente $1$ (Hardy–Littlewood — control que recuperamos).
- $R_2$: **también con exponente $1$**. Descomponiendo el canal semiprimo $q=rs$ por su
  factor primo chico $r$, cada canal lleva su propia serie singular de dos formas, y en
  el **cociente $R_2/R_1$ la serie singular se cancela** (Lema 1), dejando una suma de
  Mertens $\widetilde W(N)\sim\log\log N$. Así, **$R_2$ hereda la serie singular COMPLETA**.

El número $\beta_2\approx\tfrac12$ que reporta un ajuste log–log en rango finito **no es
un exponente** sino la **pendiente de regresión** de un objeto que no es ley de potencia;
una derivación y los datos dan
$$\boxed{\ \beta_2(N)=1-\frac{c(N)}{\langle R_2/R_1\rangle_N}+o(\cdot)\ \longrightarrow\ 1,\qquad 1-\beta_2\asymp\tfrac1{\log\log N}\ }$$

> El "½" cerca de $10^6$ es una **corrección de tamaño finito** $1/\log\log N$, no una
> deriva real. Validación no circular $(1-\beta_2)\langle R_2/R_1\rangle\approx c$ al 1%.
> La pendiente es **multi-canal**: remover el átomo $3\mid N$ baja el déficit solo ~10%
> (refuta la lectura de "un solo átomo").

**Reducción rigurosa (apéndice).** La identidad de cancelación (Lema 1) es incondicional;
una cota de 2º momento por el método del círculo — solo el arco-menor de primos de Vaughan
+ el tamaño $L^2$ trivial de la suma de semiprimos, **sin criba** — da el resultado
**lineal** $R_2(N)=\mathrm{main}(N)(1+o(1))$ para casi todo $N$, incondicional en outline.
El paso a la afirmación de **log-regresión** $\beta_2(N)\to1$ requiere un input extra
(control de cola inferior de $R_2/R_1$, más fuerte que $L^2$); lo enunciamos como tal, sin
sobre-afirmar.

## Una docena de reformulaciones, una serie singular

Goldbach se reformula de ~13 maneras; en **todas** el parámetro que gobierna es
$\mathfrak S(N)$:

| reformulación | objeto | hallazgo (todos gobernados por $\mathfrak S$) |
|---|---|---|
| frontera Chen | $R_2/R_1$, capas $\Omega(q)=k$ | $\beta_k$ cambia de signo (realce $k\le2$, supresión $k\ge3$) |
| exponente de balance | $a_{\rm Res}(N)$ (Li–Liu $1{+}a$) | obstrucción local; corr$(a_{\rm Res},\log\mathfrak S)=+0.59$ |
| energía termodinámica | $E(N)=\min(\Omega(a)+\Omega(b))$ | $E{=}2\!\iff\!$Goldbach; fusión $\beta^\star$, corr$(\beta^\star,\log\mathfrak S)=-0.92$ |
| robustez de redes | grafo $N\sim p$ | robusto-al-azar (97%) pero frágil-a-lo-modular (clase mod 3 mata 2/3) |
| mercado de liquidez | $L_G,L_C$, shock $\phi$ | Chen es liquidez de buen tiempo (sin prima de resiliencia) |
| control / RL | política $\pi(N)=p$ | info $N$ mod primos chicos: éxito 18%→39%, costo −55% |
| círculo empírico | espectro de Fourier del cometa | picos = arcos mayores; mod 6 capta 85%, → 99.9% |
| información | presupuesto de $\mathrm{Var}\log R_1$ | 99.9% comprimible (local); residuo = ruido blanco |
| geometría discreta | superficie $p+rs=N$ | nube concentrada hacia Goldbach (mediana $a=1.30$) |
| bases / acarreos | base primorial $B_y$, carries | cobertura $=\mathfrak S_y$; firma digital (+0.97 acarreos base 2) |

Paper completo (37 pp, apéndice riguroso) en
[`paper/goldbach_chen_frontier.tex`](paper/goldbach_chen_frontier.tex); las reformulaciones
transdisciplinarias se resumen en el Apéndice C y se desarrollan en el supplement
[`paper/supplement.tex`](paper/supplement.tex). Detalles y derivaciones en
[`notes/theory.md`](notes/theory.md).

## Estructura

```
proyectoGoldbach/
├── paper/goldbach_chen_frontier.tex   ← paper principal (37 pp)
├── paper/supplement.tex               ← supplement: las 7 reformulaciones transdisciplinarias
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
│   ├── robustness.py   ← robustez del grafo de Goldbach (ataques aleatorio/dirigido/modular)
│   ├── market.py       ← Goldbach como mercado: liquidez, spread de Chen, shocks de oferta
│   ├── control.py      ← Goldbach como control: políticas conscientes de residuos
│   ├── spectral.py     ← método del círculo empírico (espectro del cometa, varianza por módulo)
│   ├── information.py  ← presupuesto de información del cometa (tendencia/𝔖/irreducible)
│   ├── geometry.py     ← geometría de la superficie p+rs=N (nube 2D, espectro de exponentes)
│   ├── bases.py        ← bases primoriales (cobertura=𝔖) + geometría de acarreos (firma digital)
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
│   ├── run_energy.py   ← energía de estado fundamental aritmética + termodinámica
│   ├── run_robustness.py← robustez adversarial del grafo de Goldbach
│   ├── run_market.py   ← liquidez y resiliencia (Goldbach vs Chen)
│   ├── run_control.py  ← valor de la información aritmética para el controlador
│   ├── run_spectral.py ← método del círculo empírico
│   ├── run_information.py← presupuesto de información del cometa
│   ├── run_geometry.py ← geometría de la superficie de Chen
│   └── run_bases.py    ← cobertura primorial + exceso de acarreos
├── notes/theory.md     ← hallazgos teóricos y experimentales
├── data/   (arrays .npz y summary.json — generados)
└── figures/(PNG — generados)
```

## Reproducibilidad

```bash
python3 -m pip install -r requirements.txt   # numpy/scipy/matplotlib/sympy/pandas (Python 3.10)
python3 run_all.py            # regenera TODAS las figuras y tablas (slow ~30 min)
python3 run_all.py --fast     # salta los drivers lentos (~5 min)
python3 run_all.py --with-gurobi   # además la MIP de selección (requiere licencia Gurobi)
```

`run_all.py` contiene el **mapa figura/tabla → driver** (la tabla `DRIVERS`): p.ej. Fig. 9 →
`run_scaling.py`, Fig. 12 → `run_betalaw.py`, Fig. 24 → `run_bases.py`, CIs block-bootstrap →
`run_bootstrap.py`. Todo corre **sin Gurobi** salvo la MIP opcional (`run_mip.py`, detrás de
`--with-gurobi`). Cada `src/<mod>.py` tiene un self-test (`python3 src/<mod>.py`); las semillas
de criba y tablas derivadas están versionadas en `data/`.

**Entorno y tiempos.** Probado con **Python 3.10** y las versiones exactas de
`requirements.txt` (numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.9, sympy 1.14.0, pandas 2.3.3);
verificado clone-and-run en un *venv* limpio. Tiempos de referencia en un Intel **i7-13620H (16
núcleos, 16 GB)**: la criba por bloques escala ~lineal en `X` (~2 min por ventana de 2·10⁶ en
`X=2·10⁸`, ~10 min en `X=10⁹`); `run_all.py --fast` ~minutos, la pasada completa a 10⁹ < 1 hora.
Snapshot citable: [`10.5281/zenodo.20725701`](https://doi.org/10.5281/zenodo.20725701).

## Uso

Cada `src/*.py` tiene un self-test (`python3 src/<mod>.py`); cada figura se regenera
con su `experiments/run_*.py`. Rápidos (≲1 min) salvo donde se indica.

```bash
# --- núcleo y resultado central ---
python3 experiments/run_counts.py 2000000   # conteos, diagnósticos, figuras 01-05
python3 experiments/run_scaling.py           # β₂(X) hasta 1e9  (LENTO ~20 min)
python3 experiments/run_betalaw.py           # ley β₂=1-c/⟨R2/R1⟩  (LENTO ~5 min)
python3 experiments/run_estimator_audit.py   # estimador acumulado vs ventana + átomo 3|N
python3 experiments/run_firstmoment.py       # reducción rigurosa: respuestas A_ℓ,B_ℓ
python3 experiments/run_exceptional.py       # cota de 2º momento (conjunto excepcional)

# --- frontera Chen y diagnósticos ---
python3 experiments/run_layers.py            # capas Ω(q)=k, exponente β_k
python3 experiments/run_balance.py           # exponente de balance a_Res(N)
python3 experiments/run_continuity.py        # continuidad débil de q/N
python3 experiments/run_mip.py               # MIP: rescate + rama (requiere Gurobi)

# --- las reformulaciones ---
python3 experiments/run_energy.py            # energía termodinámica + β*
python3 experiments/run_robustness.py        # robustez del grafo
python3 experiments/run_market.py            # mercado / liquidez
python3 experiments/run_control.py           # control / valor de la información
python3 experiments/run_spectral.py          # círculo empírico (espectro)
python3 experiments/run_information.py        # presupuesto de información
python3 experiments/run_geometry.py          # geometría de p+rs=N
python3 experiments/run_bases.py             # bases primoriales + acarreos (~2.5 min)

# paper (37 pp)
cd paper && latexmk -pdf goldbach_chen_frontier.tex && latexmk -pdf supplement.tex
```

Requisitos: `numpy`, `scipy`, `pandas`, `matplotlib`, `sympy`; `gurobipy` (licencia;
solo `mip.py`/`run_mip.py`). El resto funciona sin Gurobi.

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
| `16_constante.png` | la constante $c$ del déficit: dominada por correcciones, dependiente de convención |
| `17_energia.png` | termodinámica aritmética: $Z_N(\beta)$, energía interna, calor específico, fusión $\beta^\star$ |
| `18_robustez.png` | robustez del grafo: curvas de ataque aleatorio / dirigido / modular |
| `19_mercado.png` | liquidez $L_G,L_C$ y respuesta al shock de oferta $\phi$ (Chen sin prima de resiliencia) |
| `20_control.png` | valor de la información aritmética: éxito y costo vs residuos observados |
| `21_espectral.png` | círculo empírico: espectro de Fourier del cometa, varianza captada por módulo |
| `22_informacion.png` | presupuesto de información: tendencia / $\mathfrak S$ / residuo irreducible (99.9% comprimible) |
| `23_geometria.png` | geometría de $p+rs=N$: nube 2D y espectro de exponentes $a$ |
| `24_bases.png` | cobertura primorial $=\mathfrak S_y$ y exceso de acarreos (firma digital, base 2 = Kummer) |

## Estado

- [x] Núcleo (cribas + conteos FFT) validado vs fuerza bruta; Goldbach verificado.
- [x] Serie singular: $\beta_1=1$ (control HL); $\beta_2\to1$ con déficit $1/\log\log N$ (el "½" es local).
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
- [x] Arco-mayor en detalle (§A.7): Lema 4 (Selberg–Sathé en progresiones), tracking de $A$.
- [x] **Reformulaciones**: energía termodinámica ($\beta^\star$), redes (robusto/frágil), mercado (liquidez de buen tiempo), control (valor de la info), círculo empírico, información (99.9% comprimible), geometría discreta, bases/acarreos (firma digital). Todas gobernadas por $\mathfrak S(N)$.
- [x] **Consolidación**: 19 módulos con self-test OK, 42 archivos compilan, paper 37 pp limpio, 24 figuras reproducibles.
- [ ] Demostración rigurosa COMPLETA (cerrar constantes/arco-mayor) y publicación.
- [ ] Column generation del MIP infinito (pricing real).
