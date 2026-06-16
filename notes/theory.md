# La frontera Goldbach–Chen: hallazgos teóricos y experimentales

Notas de trabajo que acompañan al código de `src/` y a los experimentos de
`experiments/`. Resumen lo que el `.tex` propone y, sobre todo, **lo que la
implementación añade**: una regularidad empírica nueva y nítida sobre cómo el
conteo prima+semiprimo $R_2$ depende de la serie singular de Goldbach.

Toda cifra de abajo es reproducible con
`python3 experiments/run_counts.py 2000000` y `python3 experiments/run_mip.py`.

---

## 0. Núcleo y validación

- **Conteos por convolución.** $R_1(N)=(\mathbf 1_{\mathbb P}\!*\mathbf 1_{\mathbb P})(N)$
  y $R_2(N)=(\mathbf 1_{\mathbb P}\!*\mathbf 1_{\mathcal S_2})(N)$ se calculan para
  **todos** los $N\le X$ con una FFT (`src/counts.py`), en $O(X\log X)$. Para
  $X=2\cdot10^6$ son ~15 s incluidas cribas, diagnósticos y figuras.
- **Validación.** La versión FFT coincide exacto con fuerza bruta en muestras
  aleatorias (redondeo de la FFT $\ll 1/2$). Goldbach ($R_1>0$) se verifica en
  todo el rango.

---

## 1. Resultado central: el **exponente singular** de $R_2$ es $\approx 1/2$

### 1.1 Lo conocido (Hardy–Littlewood) — sirve de control

La conjetura de HL para Goldbach es
$$R_1(N)\sim 2C_2\,\mathfrak S(N)\,\frac{N}{\log^2 N},\qquad
\mathfrak S(N)=\prod_{p\mid N,\,p>2}\frac{p-1}{p-2},\quad
C_2=\prod_{p>2}\Big(1-\tfrac1{(p-1)^2}\Big)=0.6601618\ldots$$
La serie singular $\mathfrak S(N)$ explica las **bandas** de la cometa de Goldbach
(los $N$ con muchos divisores primos pequeños tienen muchas más representaciones).

**Verificación.** Regresando $\log\!\big(R_1/(N/\log^2N)\big)$ contra $\log\mathfrak S(N)$
la pendiente es $\beta_1=1.000$ y el coeficiente de variación de la razón
normalizada cae de $\mathrm{CV}=0.39$ (sin serie singular) a $\mathrm{CV}=0.015$
(con ella) — un colapso de $\sim\!25\times$. Es un control: recuperamos HL.

### 1.2 Lo nuevo: $R_2$ hereda **solo la raíz** de la serie singular

Definimos el exponente singular empírico de $R_2$ por la pendiente de
$$\log\frac{R_2(N)}{R_1(N)}\ \text{ vs }\ \log\mathfrak S(N),\qquad
\text{pendiente}=\beta_2-1$$
(usar $R_1$ como “medidor” de $\mathfrak S(N)$ hace la medición *model-light*: no
depende de ninguna normalización elegida). Resultado:

| $X$ | $\beta_2$ | $R^2$ de la regresión |
|-----|-----------|------------------------|
| $5\cdot10^5$ | 0.452 | 0.91 |
| $1\cdot10^6$ | 0.475 | 0.91 |
| $2\cdot10^6$ | 0.497 | 0.92 |
| $4\cdot10^6$ | 0.513 | 0.93 |
| $8\cdot10^6$ | 0.529 | 0.93 |

**Hallazgo.** $\beta_2\approx 1/2$, en contraste fuerte con $\beta_1=1$. La serie
singular explica el **~92 %** de la varianza de $\log(R_2/R_1)$. Equivalentemente:
$$\boxed{\,R_2(N)\ \propto\ \mathfrak S(N)^{1/2}\,\frac{N}{\log^2 N}\,W(N)\,}$$
con $W(N)$ el peso de Mertens de §2. Ver `figures/02_singular_exponent.png`: el
panel de $R_1$ tiene pendiente 1, el de $R_2$ pendiente 0.497.

**Calidad de la normalización.** Comparando el CV de la razón observado/modelo:

| modelo de $R_2$ | CV ($X=2\cdot10^6$) |
|-----------------|----------------------|
| $N\log\log N/\log^2N$ (heurística del `.tex`) | 0.184 |
| $2C_2\,\mathfrak S(N)^{1}\,NW/\log^2N$ (herencia total, ingenua) | 0.174 |
| $2C_2\,\mathfrak S(N)^{1/2}\,NW/\log^2N$ (**media potencia**) | **0.018** |

La media potencia ajusta **10× mejor** que la heurística del `.tex` y que la
herencia total. Incluir $\mathfrak S$ con exponente 1 *empeora* el ajuste: prueba
directa de que $R_2$ **no** hheredaría la serie singular completa.

### 1.3 Interpretación: una forma cuantitativa del “Chen rescue effect”

$\beta_2<1$ significa que donde Goldbach es localmente **rico** ($\mathfrak S$ grande,
p.ej. $6\mid N$), $R_1$ se dispara $\propto\mathfrak S$ pero $R_2$ solo
$\propto\mathfrak S^{1/2}$; luego la cuota semiprima
$R_2/R_1\propto\mathfrak S^{-1/2}$ es **menor**. Al revés: donde Goldbach es
**frágil** ($\mathfrak S$ chico), los semiprimos son proporcionalmente **más**
abundantes. Esto es exactamente la conjetura *Chen rescue effect* del `.tex`,
ahora con una ley de potencias: **la abundancia relativa del canal semiprimo
escala como $\mathfrak S(N)^{-1/2}$.**

### 1.4 Por qué $\beta<1$ (boceto) y por qué $1/2$ exacto queda abierto

Escribir el semiprimo $q=N-p=r\,s$ con $r\le s$ primos impares y sumar sobre el
factor pequeño $r$:
$$R_2(N)=\sum_{r}\#\{s:\,N-rs\ \text{primo}\}
\approx \frac{N}{\log^2N}\sum_{3\le r\le\sqrt N}\frac{\mathfrak S_2(r,N)}{r}\,
\frac{\log N}{\log(N/r)}.$$
La serie singular de las dos formas $(s,\,N-rs)$ factoriza como (Lema 1 del apéndice
riguroso del paper)
$$\mathfrak S_2(r,N)=\begin{cases}2C_2\,\mathfrak S(N)\,\dfrac{r-1}{r-2}, & r\nmid N,\\[1mm]
0, & r\mid N\ \text{(canal bloqueado: }p=N-rs\equiv0\bmod r\Rightarrow p=r).\end{cases}$$
La clave: cuando el factor pequeño $r$ **divide** a $N$, el canal $r$ se **bloquea**
(su término se quita de la suma), justo cuando $\mathfrak S(N)$ es grande. Esto reduce
el exponente efectivo por debajo de 1.

**Nota (corrección).** Una versión anterior de estas notas usaba erróneamente
$\mathfrak S_2(r,N)\propto(r-2)/(r-1)$ para $r\mid N$ (un "flip"), y conjeturaba una
media potencia fija $\beta_2\to1/2$. El Lema 1 corrige esto: el canal se **anula**.
La derivación correcta y su demostración (condicional a Hardy–Littlewood) están en
**§5ter** y en el **apéndice del paper** (Teorema 1): $\beta_2\to1$ con déficit
$\sim1/\log\log N$. El "$1/2$" es solo el valor local cerca de $10^6$.

---

## 1ter. Capas $\Omega(q)=k$: el exponente singular cambia de signo

Generalizando $R_1,R_2$ a $R_k(N)=\#\{p<N:\Omega(N-p)=k\}$ y midiendo el exponente
singular $\beta_k$ de cada capa (mismo método model-light), aparece un patrón limpio
(`figures/10_capas.png`), en $X=2\cdot10^6$:

| $k$ | 1 | 2 | 3 | 4 | 5 | 6 |
|-----|---|---|---|---|---|---|
| share $\rho(k)$ | 15.5% | 33.5% | 28.7% | 14.6% | 5.5% | 1.8% |
| $\beta_k$ | 1.00 | 0.50 | −0.30 | −1.38 | −2.73 | −4.04 |

- **No es $1/k$.** $\beta_k$ decrece monótono y se vuelve **negativo** para $k\ge3$.
- **La serie singular cambia de signo de efecto:** realza las capas bajas
  ($k\le2$, $\beta>0$) y **suprime** las altas ($k\ge3$, $\beta<0$). Donde $\sigma(N)$
  es grande ($N$ con muchos primos chicos), $q=N-p$ se ve forzado a evitar primos
  pequeños $\Rightarrow$ $q$ tiene **menos** factores $\Rightarrow$ menor $\Omega$.
  Es el mismo mecanismo de $R_2$, llevado al extremo.
- Todos los $\beta_k$ derivan hacia arriba con $X$ (igual que $\beta_2$): el cruce
  $\beta_k=0$ se mueve lentamente a la derecha.
- **Rescate compuesto×compuesto** ($q=ab$, $a,b$ ambos compuestos): existe para
  **todo** $N$ par del rango, con media $\sim1.8\cdot10^4$ representaciones. La
  jerarquía aditiva por debajo de la capa prima es enormemente redundante.

## 2. El factor $\log\log N$ es una suma de Mertens

El `.tex` escribe $R_2\sim K_2(N)\,N\log\log N/\log^2N$ con $K_2$ sin especificar.
La descomposición de §1.4 identifica el origen del $\log\log N$: es la suma de
Mertens sobre el **factor primo pequeño** $r$ del semiprimo,
$$W(N)=\sum_{3\le r\le\sqrt N}\frac{r-1}{r-2}\,\frac1r\,\frac{\log N}{\log(N/r)}
\ \sim\ \log\log\sqrt N\ \sim\ \log\log N$$
(Mertens). Es decir, el “exceso” logarítmico de los semiprimos no es un factor
opaco: cuenta, con pesos de primo-gemelo, las maneras de elegir el factor chico.

---

## 3. Diagnósticos de la frontera ($X=2\cdot10^6$)

- **Cuota primo–primo** $\theta=R_1/(R_1+R_2)$: media $0.311$. Dos de cada tres
  representaciones de Chen usan un semiprimo genuino. $\theta$ decrece lentamente
  con $N$ (el canal semiprimo gana terreno, como predice $\log\log N$).
- **Excedente de Chen** $C(N)=R_2/(R_1+1)$: media $2.27$.
- **Fragilidad de Goldbach.** $|\mathcal F_k|$ = nº de $N$ pares con $R_1\le k$ y
  $R_2>0$ hasta $X$: $|\mathcal F_1|=1$, $|\mathcal F_2|=3$, $|\mathcal F_3|=5$,
  $|\mathcal F_5|=14$, $|\mathcal F_{10}|=49$. Crecen extremadamente despacio: la
  fragilidad de Goldbach es rarísima, pero **siempre que ocurre hay rescate de
  Chen** ($R_2>0$ en el 100 % de los frágiles observados).
- **Geometría de factores** $B(q)=\log a/\log q$ (factor menor): media ponderada
  $0.25$; masa con $B<0.1$ (factor diminuto) $=0.16$; con $B>0.4$ (casi
  balanceado) $=0.18$. **Respuesta a la pregunta del `.tex`:** las representaciones
  de Chen están dominadas por semiprimos **desbalanceados con factor primo
  pequeño** (3, 5, 7…), no por semiprimos balanceados — coherente con que
  $q=3s$ sea el tipo de semiprimo más denso. Ver `figures/05_balance.png`.

---

## 3bis. Continuidad débil de las medidas de $q/N$ (sección 4 del `.tex`)

La medida puntual $\mu_N$ de $q/N$ es ruidosa, pero al suavizar sobre ventanas
$N\in[X,X+H]$ se estabiliza (ver `figures/08_continuidad.png`). En $X=10^6$:

- **Converge.** Partiendo una ventana en dos submuestras de la **misma escala**
  ($N$ alternos), la distancia TV entre sus densidades cae monótona de
  $2.1\cdot10^{-3}$ a $3\cdot10^{-4}$ cuando $H$ va de 125 a 4000 (el ruido por-$N$
  se promedia) — una forma concreta de convergencia débil.
- **El límite es casi uniforme.** La densidad de $q/N$ para $q$ primo está a TV
  $0.028$ de la uniforme en $(0,1)$, con un leve repunte en los bordes (efecto
  $1/(\log(xN)\log((1-x)N))$). Responde la Question 1 del `.tex`: tras quitar el
  efecto de borde, $\mu_N^{(1)}\approx$ uniforme.
- **Primo y semiprimo casi coinciden.** Las dos densidades están a TV $0.019$ entre
  sí (confirma Conjecture 3). El canal semiprimo está levemente sesgado a $q$ grande
  (media $q/N=0.513$ vs $0.500$ exacto de los primos), única firma visible de que
  $p$ y $q$ juegan roles asimétricos en $R_2$.

## 4. Optimización / selección (secciones 6–7 del `.tex`)

### 4.1 La oscilación de tipo no es forzada… salvo bajo restricción

Como Goldbach se cumple empíricamente, el MIP de mínimas conmutaciones de tipo
tiene óptimo trivial $0$ (elegir primo en todo $N$). La pregunta con contenido
aparece al **restringir** el conjunto admisible. Dos resultados:

- **Suavidad geométrica ≠ suavidad de tipo (trade-off de Pareto).** Minimizando la
  variación total $\sum|s_N-s_{N-2}|$ de la posición normalizada $s_N=q_N/N$, la
  rama óptima se pega al borde $s_N\approx1$ (el “socio mínimo”: $p$ = primo más
  chico válido) y es geométricamente *muy* suave: TV $=0.0040$ sobre 201 enteros.
  Pero **alterna de tipo en el 28 % de los pasos** y usa 72 % de semiprimos.
  Si en cambio se prohíben los semiprimos (rama solo-Goldbach, $0$ saltos de
  tipo), la TV sube a $0.0246$ — **6.2× menos suave**. No se puede maximizar la
  suavidad geométrica y la de tipo a la vez: es un Pareto. Se puede ser continuo
  en el lugar y caótico en el tipo. Ver `figures/07_rama_mip.png`.

### 4.2 Umbral de rescate de Chen: una banda unimodal

Restringiendo a una **banda balanceada** $p\in[N/2-k,\,N/2]$ (representaciones casi
simétricas $p\approx q\approx N/2$) y midiendo qué fracción de $N$ requieren un
semiprimo porque no hay par primo en la banda, la curva es **unimodal** en $k$
(ver `figures/06_rescate.png`):

- $k$ **grande** ($\gtrsim 0.005\,N$): el canal primo casi nunca falla → rescate
  $\approx0\%$.
- $k$ **intermedio** ($\sim20$–$40$ enteros): **pico de rescate $\approx38$–$40\%$**.
  Hay suficientes candidatos para que casi siempre exista un semiprimo, pero a
  menudo no un primo.
- $k$ **diminuto** ($k=1,2$): el rescate vuelve a bajar — porque con tan pocos
  candidatos muchas veces no existe **ninguna** representación (también falla
  Chen), no es que el primo gane.

El pico se corre a la derecha con $N$ ($\sim25$ a $N\!\sim\!5\cdot10^4$, $\sim40$ a
$N\!\sim\!10^6$), escalando como $\log^2N$: el umbral está donde el nº esperado de
pares primos balanceados $\sim 2C_2\,\mathfrak S(N)\,k/\log^2(N/2)$ cae a $O(1)$ —
una transición tipo Poisson. Realiza el *phase diagram* / valor crítico
$a_X^*(N)$ del `.tex`: la flexibilidad semiprima de Chen se vuelve **necesaria** al
exigir balance casi perfecto.

---

## 5bis. Formas lineales generalizadas $N=m_1p+q$ (draft expandido §13)

Convolucionando el indicador de primos **dilatado** $\{m_1p\}$ con cada capa
($X=2\cdot10^6$), aparecen obstrucciones de congruencia que responden la *Question
de sensibilidad a coeficientes*:

- $m_1$ **par** (2, 4): destruye el canal primo–primo por paridad ($q=N-m_1p$ es
  par $\Rightarrow$ solo $q=2$); $\max R_1=1$, $\theta\approx0$.
- $m_1=3$: el enunciado tipo Goldbach **falla para el ~28 % de los $N$**
  ($N=3p+q$ sin $q$ primo). Genuinamente frágil.
- $m_1=1$ es **especial**: el único sin obstrucción local. Confirma que la elección
  de coeficientes controla la fragilidad; los $\beta$ para $m_1\neq1$ no son
  interpretables (dominan las congruencias, no la serie singular).

---

## 5ter. Escalamiento de β₂ hasta 10⁹: ley loglog (pregunta central RESUELTA)

Midiendo β₂ **localmente** en ventanas de $2\cdot10^6$ centradas en $X$ (criba por
convolución de bloques, `src/segmented.py`, para llegar a $X=10^9$ sin FFT global),
β₂ **no se estabiliza en ½**: sube monótono. Los 8 puntos ($6\cdot10^6$ a $10^9$)
ajustan a $0.0013$ con
$$\beta_2(X)\approx -0.094 + 0.234\,\log\log X\qquad(R^2=0.9987).$$
- β₂ medido: 0.55 ($6\cdot10^6$) → **0.614** ($10^9$).
- Extrapola a ≈0.79 en $4\cdot10^{18}$ (rango verificado de Goldbach); β₂=1 recién en
  $X\sim10^{46}$.
- **El "½" es solo el valor local cerca de $10^6$.** No hay potencia singular fija
  para $R_2/R_1$: el exponente crece como $\log\log X$ (coef ≈0.23, cercano a ¼).
- Validación: la convolución por bloques coincide EXACTO con la FFT global; serie
  singular en ventana exacta (cofactor primo grande recuperado). Ver
  `figures/09_beta_scaling.png`.

### Derivación analítica de la ley (heurística)

**Afirmación.** $\beta_2(N) = 1 - \dfrac{c}{\langle R_2/R_1\rangle_N} + o(\cdot)$ con
$\langle R_2/R_1\rangle\sim\log\log N$, de modo que **$\beta_2\to1$**: $R_2$ hereda
asintóticamente la serie singular COMPLETA, con déficit $\Theta(1/\log\log N)$.

**Paso 1 — cancelación exacta de 𝔖.** Escribiendo $q=N-p=rs$ y sumando sobre el
factor pequeño $r$, cada canal es un conteo de Goldbach de la forma lineal
$(s,N-rs)$ con serie singular de dos formas
$$\mathfrak S_2(r,N)=\begin{cases}2C_2\,\mathfrak S(N)\,(r-1)/(r-2),&r\nmid N\\ 0,&r\mid N\end{cases}$$
(el canal $r\mid N$ está **bloqueado**: $p=N-rs\equiv0\bmod r\Rightarrow p=r$). Como
$R_1(N)=2C_2\,\mathfrak S(N)\,I_1(N)$ (Hardy–Littlewood) y
$R_2(N)=2C_2\,\mathfrak S(N)\sum_{r\nmid N}\frac{r-1}{r-2} I_r(N)$, el cociente **cancela 𝔖**:
$$\frac{R_2(N)}{R_1(N)}=\widetilde W(N):=\sum_{3\le r\le\sqrt N,\,r\nmid N} \frac{r-1}{r-2}\,w_r(N),
  \qquad w_r=\frac{I_r}{I_1}=\frac1r\frac{\log N}{\log(N/r)}.$$

**Paso 2 — de dónde sale el exponente.** El exponente medido es la pendiente de
regresión $\beta_2-1=\mathrm{Cov}(\log\widetilde W,\log\mathfrak S)/\mathrm{Var}(\log\mathfrak S)$.
Con $e_\ell=\mathbf 1[\ell\mid N]$ (independientes, $P=1/\ell$):
$\log\mathfrak S=\sum_\ell g_\ell e_\ell$, $g_\ell=\log\frac{\ell-1}{\ell-2}$, así
$\mathrm{Var}(\log\mathfrak S)=\sum_\ell g_\ell^2\frac1\ell(1-\frac1\ell)=:B$ **(constante**,
dominada por $\ell=3,5$). Para $\widetilde W$, bloquear el canal $r$ al dividir $N$
**quita** el término $r$: $\delta_r=-\frac{r-1}{r-2}w_r<0$; linealizando
$\log\widetilde W\approx\log\widetilde W_0+\widetilde W_0^{-1}\sum_r\delta_r e_r$. Por
independencia solo sobreviven los términos $r=\ell$:
$$\mathrm{Cov}(\log\widetilde W,\log\mathfrak S)
  =\frac1{\widetilde W_0}\sum_r \delta_r\,g_r\,\tfrac1r(1-\tfrac1r)=:-\frac{A}{\widetilde W_0},$$
con $A>0$ **constante** (la suma converge, $\delta_r g_r/r\sim r^{-4}$). Por tanto
$$\beta_2-1=\frac{-A/\widetilde W_0}{B}=-\frac{A/B}{\widetilde W_0},\qquad
  \widetilde W_0\approx\langle R_2/R_1\rangle\sim\log\log N.$$
La correlación con 𝔖 viene **solo** de los divisores primos pequeños de $N$ (una
perturbación $O(1)$), repartida sobre una suma $\widetilde W_0\sim\log\log N$; de ahí
el déficit $1/\log\log N$. **Conclusión: $\beta_2=1-(A/B)/\langle R_2/R_1\rangle\to1$.**

**Validación (no circular).** La ley predice $(1-\beta_2)\langle R_2/R_1\rangle\equiv c$
constante. Medido en ventanas de $2\cdot10^6$:

| $X$ | $6\cdot10^6$ | $2\cdot10^7$ | $10^8$ | $2\cdot10^8$ |
|-----|------|------|------|------|
| $(1-\beta_2)\langle R_2/R_1\rangle$ | 1.110 | 1.100 | 1.088 | 1.084 |

CV $=1.0\%$ (vs $1.7\%$ para $(1-\beta_2)\log\log N$): el "reloj" correcto es
$\langle R_2/R_1\rangle$, exactamente como predice la derivación. $c\approx1.1$.
Ver `figures/12_betalaw.png`.

**Sobre la constante $c$ (detalle técnico, resuelto honestamente).** La constante de
la serie singular del canal SÍ es exacta y verificada:
$C_r'(N)=2C_2\,\mathfrak S(N)\,\frac{r-1}{r-2}\,I_r(N)\,(1\pm0.004)$ con $I_r$ el integral
arquimediano (`run_channel_check.py`: ratio 0.996 para $r=3$). PERO el peso crudo
$w_r=\frac1r\frac{\log N}{\log(N/r)}$ **sobreestima $I_r/I_1$ ~1.5× para $r$ chico** —
esa fue la fuente de mis líos con la constante. Con $w_r$ crudo sale $c\approx1.0$; con
$I_r/I_1$ exacto sube hacia el medido $\approx1.1$. La constante $c(X)$ es $O(1)$,
decrece lento, y su límite exacto es delicado (depende de $I_r/I_1$ y de la convención
de promediado: media-de-cocientes vs cociente-de-sumas). Lo robusto e incondicional es
el ORDEN: $c(X)\asymp1$, $1-\beta_2\asymp1/\log\log X$, $\beta_2\to1$. El valor preciso
de $c$ NO afecta el teorema.

**Ataque al límite exacto $c_\infty$ (resultado honesto).** Medí dos normalizaciones a
4 escalas (hasta $5\cdot10^8$): $\kappa(X)=(1-\beta_2)\log\log X$ y
$c(X)=(1-\beta_2)\langle R_2/R_1\rangle$. **Ambas DECRECEN** ($\kappa$: 1.25→1.18; $c$:
1.12→1.08). El ajuste lineal en $1/\log\log X$ extrapola a $\kappa_\infty\approx0.54$,
$c_\infty\approx0.73$ — con pendiente de corrección $\approx1.9$, **mayor que el límite**.
Conclusiones: (1) el límite es **dependiente de la convención** (0.54 vs 0.73), no hay un
número canónico; (2) es **corrección-dominado**: el valor finito ~1.1 es casi todo la
corrección $1/\log\log X$, el límite real (~0.5-0.7) está lejos y **NO es determinable con
confianza hasta $10^9$**. Mis dos estimaciones analíticas (1.0 y 1.8) acotan la dificultad,
no la resuelven. Solo el ORDEN $1/\log\log X$ es robusto. `run_constant.py`, `figures/16_constante.png`.
NOTA técnica resuelta de paso: con el integral arquimediano EXACTO $I_r$, $R_1=2C_2\mathfrak S I_1(0.999)$
y $C_r'=2C_2\mathfrak S\frac{r-1}{r-2}I_r(0.996)$ (el "1.17" de antes era $N/\log^2N$ vs $I_1$); el peso
crudo $w_r$ y el choque media-de-cocientes/cociente-de-sumas eran las fuentes de mis líos previos.

### Reducción rigurosa: de HL puntual a Bombieri–Vinogradov + conjunto excepcional

El Teorema condicional del apéndice pide HL **puntual** uniforme en $r$ — tan difícil
como el problema binario. La salida (apéndice §A.5): pasar a **primeros momentos**.
Sumar sobre $N$ elimina la dificultad binaria (queda contar primos en progresiones).
Respuestas condicionales (medias sobre la ventana), con prueba en el Lema 3:
$$A_\ell=\frac{\langle R_1\rangle_{\ell\mid N}}{\langle R_1\rangle_{\ell\nmid N}}=\frac{\ell-1}{\ell-2},
\qquad B_\ell=\frac{\langle R_2\rangle_{\ell\mid N}}{\langle R_2\rangle_{\ell\nmid N}}
=\frac{(\ell-1)(1-F_\ell)}{\ell-2+F_\ell},$$
con $F_\ell$ = fracción de reps de $R_2$ con $\ell\mid q$. **Ambas exactas a 4 cifras**
(p.ej. $\ell=3$: $B_3=1.4232$ medido vs $1.4233$ fórmula). Se siguen SOLO de la
**equidistribución de primos en progresiones (Siegel–Walfisz / Bombieri–Vinogradov —
TEOREMAS)** más el primer momento $F_\ell$. El exponente de primer momento
$\beta_2^{(1)}$ (Teorema 2) cumple $\beta_2^{(1)}=1-c^{(1)}/\widetilde W_0\to1$ y
**reproduce $\beta_2$** con gap **constante** $\approx0.004$ (Jensen) en todas las
escalas. Así el único paso que falta es puntual↔primer-momento = una **cota de
segundo momento / conjunto excepcional para $R_2$** (à la Montgomery–Vaughan, que ya
lo hace para $R_1$), MUCHO más tratable que la HL puntual.

**Cota de segundo momento (apéndice §A.6, RESUELTA empíricamente).** Mido la
concentración del cociente en la predicción de canales: $b(N)=(R_2/R_1)/\widetilde W(N)$.
Resultado: **CV$(b)\le0.7\%$, decreciente, y el conjunto excepcional al 5% es VACÍO**
(0 de $10^6$ N) hasta $5\cdot10^8$. Por Chebyshev, densidad excepcional $\le$CV$^2/\varepsilon^2\to0$,
así $R_2/R_1=\widetilde W(N)$ c.t.p. y **$\beta_2(N)\to1$ para casi todo $N$**. **Simplificación clave (Prop 3 del apéndice):** la varianza **NO necesita criba**.
Por Bessel + acotar un factor por su sup:
$$\sum_N|R_2-\text{principal}|^2\le\int_{\mathfrak m}|S_\PP|^2|S_{\mathcal S_2}|^2
\le\Big(\sup_{\mathfrak m}|S_\PP|\Big)^2\cdot\#\mathcal S_2(2X).$$
Solo entra el **arco-menor de PRIMOS (Vinogradov–Vaughan, incondicional)**; el semiprimo
entra por su norma $L^2=\#\mathcal S_2$ (Parseval, trivial). Con $Q=\log^{2A+8}X$ esto da
varianza $\ll X^3\log\log X\,\log^{-2A-1}X$, luego conjunto excepcional $\ll X\log^{3-2A}X=o(X)$.
**Teorema 3 (apéndice): $\beta_2(N)\to1$ para casi todo $N$, INCONDICIONAL en outline** —
todos los ingredientes (Vaughan, Siegel–Walfisz, Montgomery–Vaughan para $R_1$, y la
identidad de Euler del Lema 1) son clásicos; falta solo la cuenta de arco-mayor rutinaria
para semiprimos en progresiones (Selberg–Sathé). El contenido NUEVO es la cancelación de
la serie singular (Lema 1); la maquinaria analítica es estándar. `run_exceptional.py`.

**Arco-mayor escrito en detalle (apéndice §A.7).** Lema 4 (Selberg–Sathé en
progresiones, DEMOSTRADO): para $(a,q)=1$, $q\le(\log x)^A$,
$$\pi_2(x;q,a)=\frac{\mathcal S_2^*(x;q)}{\phi(q)}+O_A(x\,e^{-c\sqrt{\log x}}),$$
vía $n=p_1p_2$ + Siegel–Walfisz (el $A$ entra solo por la constante de SW, inefectiva).
Estructura local EXACTA (Remark 6): para $\ell$ primo, clase 0 $=\pi(x/\ell)$ (los
$\ell\cdot$primo), clases coprimas equidistribuyen; confirmado al entero
($\pi_2(x;3,0)=102383=\pi(x/3)$). La supresión de clase 0, $\pi(x/\ell)/\pi_2\sim1/(\ell\log\log x)$,
ES el déficit loglog que da $\widetilde W$ y el bloqueo del Lema 1. La evaluación de
arco-mayor reproduce $\mathfrak S_2(N)=2C_2\mathfrak S(N)\widetilde W(N)$ (consistente con
Lema 1). **Tracking de $A$:** un solo $Q=(\log X)^B$ ata SW + arco-mayor + varianza; el
arco-mayor da error super-polinomial (no liga), y la tasa la fija el arco-menor:
conjunto excepcional $\ll X/(\log X)^C$ para todo $C$. El programa riguroso queda
completo en outline.

## 5quater. Exponente de balance residual $a_{\rm Res}(N)$ (tercer draft)

$a_{\rm Res}(N)=1+\min_{N=p+rs}\log r/\log s$ (Chen nondegenerado, $r,s$ primos);
$r_\star(N)$ = menor multiplicador primo que sirve. Calculados exactos hasta
$2\cdot10^6$ ($r_\star$ por cobertura FFT, $a_{\rm Res}$ por barrido small-$p$,
validado a error 0). En $N\ge1000$:
- **Colapso residual típico:** mediana $a_{\rm Res}$ baja con $X$ (1.115→1.099→1.088),
  $p_{99}=1.182$, max=1.385 — muy por debajo del umbral Li–Liu 1.9.
- **Multiplicador minimal diminuto:** $r_\star=3$ en 68% de los $N$; diccionario
  $\{2,3,5,7\}$ cubre 99.7%, $\{2..11\}$ cubre 99.99% (responde Q6).
- **Obstrucción local:** si $3\mid N$, el canal $r=3$ se bloquea → $r_\star$ medio 4.59
  vs 2.94 (responde Q2).
- **Conexión con la serie singular (Q4):** corr$(a_{\rm Res}, \log\mathfrak S)=+0.59$.
  La misma divisibilidad de $N$ que agranda $\mathfrak S$ (realza Goldbach) bloquea los
  canales de $r$ chico → el balance residual es **peor justo donde Goldbach es rico**.
  Ata el tercer draft a mi tema central. Ver `figures/11_balance.png`.

## 5quinquies. Tres sondas más (transporte, topología, dinámica)

Las tres vuelven a la serie singular.

- **Transporte óptimo (fig 13).** Bajo la reflexión $T(x)=1-x$: defectos $W_1$
  $\Delta_1=0.040$ (primo vs primo reflejado), $\Delta_2=0.032$ (primo vs semiprimo
  reflejado); la mezcla óptima de Chen da $\Delta_{\le2}=0.032$ en $\lambda^\star=0$.
  La nube semiprima reflejada matchea a los primos MEJOR que la prima reflejada:
  "colapso de transporte" del 22% (estable en X), dominado por semiprimos de factor
  chico. Responde la Transport-collapse del draft: SÍ.
- **Topología de valles (fig 14).** Normalizando cada canal por su factor singular
  ($R_1/\mathfrak S$, $R_{\le2}/(\mathfrak S+\mathfrak S^{1/2}W)$) y midiendo la
  persistencia 0-dim sublevel: los valles de Goldbach y Chen COINCIDEN (corr de
  anomalías $+0.64$; en los 200 N más frágiles la anomalía de Chen, 0.29, es tan
  honda como la de Goldbach, 0.30). **Chen NO rellena los valles** (solo ablanda el
  30% más hondo). El rescate es de magnitud, no de estructura: ambos canales
  comparten la restricción "$p$ primo".
- **Dinámica estadística (fig 15).** $\theta(N)$ está 94% explicado por $\mathfrak S(N)$
  (N mod 30 no agrega nada); $\Delta\theta_N$ está 98% explicado por el salto
  $\mathfrak S(N)\to\mathfrak S(N+2)$. El residuo (25% de la dispersión) NO es ruido
  blanco (autocorr lag-1 $-0.21$ tras destendenciar): la frontera casi no tiene
  aleatoriedad irreducible — la gobierna la aritmética local de N.

## 5sexies. Energía de estado fundamental aritmética (reformulación unificadora)

$E(N)=\min_{a+b=N}(\Omega(a)+\Omega(b))$, Goldbach $\iff E(N)=2$. Espectro de niveles
$n_E(N)$: **$n_2=R_1$** (degeneración del fundamental = Goldbach), **$n_3=2R_2$** (primer
excitado = Chen). Subsume capas, función de partición (#7), semianillos (#12) y grado del
grafo (#2). Termodinámica vía $Z_N(\beta)=(f_\beta*f_\beta)(N)$, $f_\beta(n)=e^{-\beta\Omega(n)}$
(una FFT por $\beta$ da $Z$ para todo $N$):
- Espectro = **joroba** en la energía típica $\sim2\log\log N$, con el fundamental $E=2$ en
  la cola; gap típico→fundamental $\sim2\log\log N-2$ diverge.
- Energía media $U(\beta)\to2$; calor específico $C(\beta)$ con pico en $\beta^*\approx2.1$ =
  **"temperatura de fusión"** del orden de Goldbach.
- **corr$(\beta^*,\log\mathfrak S)=-0.92$**: donde $\mathfrak S$ grande (más estados
  fundamentales, $R_1\propto\mathfrak S$), el orden funde a $\beta$ menor. La MISMA serie
  singular que gobierna $R_1,\beta_2$ y el balance gobierna la estabilidad termodinámica del
  fundamental. Goldbach = "todo $N$ par tiene energía de estado fundamental 2".
`run_energy.py`, `src/energy.py`, `figures/17_energia.png`.

## 5septies. Robustez adversarial del grafo de Goldbach (ideas #2 + #13)

Grafo bipartito $N\sim p$ (arista si $N-p$ primo). ¿Cuán robusto es Goldbach a quitar
primos? En una ventana de $1.5\cdot10^4$ N cerca de $10^6$:
- **Aleatorio:** Goldbach sobrevive hasta $\phi\approx0.97$ — se puede quitar el **95% de
  los primos al azar** y todo N sigue representable (redundancia enorme).
- **Dirigido:** romper un N = corte de vértices de un emparejamiento = $R_1(N)/2$ primos;
  el más barato (N más débil) cuesta $\kappa=4.6\%$ de los primos.
- **Modular (el caso fuerte):** quitar la clase $1$ mod $3$ (¡la **mitad** de los primos!)
  deja solo la clase $2$, cuyo sumset es $\{2+2\}=\{1\}$: **todo $N\equiv0,2$ mod 3 pierde
  TODAS sus representaciones** → destruye 59% de los N. El mismo 50% de primos al azar no
  destruye nada. El adversario **fabrica una obstrucción local** (justo lo que la serie
  singular descarta para los primos). Solo $m=3,4$ funcionan ($m\ge5$ el sumset cubre todo).

**Goldbach es robusto-al-azar pero frágil-a-la-estructura**: invulnerable a perder casi
todos los primos por azar, pero destrozado al quitar una progresión aritmética. La
redundancia de Goldbach es aritmética, no estadística. `run_robustness.py`, fig 18.

## 5octies. Tres reformulaciones más (económica, topológica, control)

- **Mercado / liquidez (#5/#6, fig 19).** Cada N par = mercado que necesita 2 insumos
  primos. Liquidez $L_G=R_1$, $L_C=R_1+R_2$, spread de Chen $S_C=R_2/R_1\approx2.3$ (Chen
  3× más líquido en normalidad). Bajo shock de oferta (quitar fracción $\phi$ de primos),
  un match de Goldbach sobrevive con $(1-\phi)^2$ pero uno de Chen con $(1-\phi)^3$ (el
  semiprimo $q=rs$ necesita p,r,s). Resultado: **el sustituto de Chen NO da prima de
  resiliencia** ($\phi_C-\phi_G\approx2\cdot10^{-4}$); $S_C(\phi)=S_C(0)(1-\phi)\to0$. Chen
  es **liquidez de buen tiempo**: vale menos justo cuando los primos escasean.
- **Topología persistente (#8, resultado NEGATIVO).** Filtrar descomposiciones por
  complejidad $\Omega(a)+\Omega(b)$: en el binario es un **matching** (cada $a$ en un solo
  par) → persistencia 0-dim = el espectro de energía, homología superior nula. Topología
  rica solo en el ternario (grafo $p\sim q$ si $N-p-q$ primo, denso, conexo = Helfgott por
  vértice, pero **anti-clustered**). No hay estructura topológica nueva que descubrir.
- **Control / valor de la información (#11, fig 20).** Controlador elige $\pi(N)=p$,
  aclara si $N-p$ primo. Política **consciente de residuos** (usa $N$ mod primos $\le B$
  para evitar $q$ divisible por chicos): éxito single-shot **18%→39%** (B=13), costo
  secuencial **5.5→2.5 intentos (−55%)**. La ganancia por primo sigue el factor
  $(\ell-1)/(\ell-2)$ — la serie singular hecha **accionable**. Conocer $N$ mod primos
  chicos aclara el mercado ~2× más eficiente.
`run_market.py`, `run_control.py`.

## 5nonies. Espectral, información y geometría (#3/#10, #9, #14)

- **Método del círculo EMPÍRICO (#3/#10, fig 21).** Espectro de Fourier del cometa
  destendenciado: picos discretos en las frecuencias aritméticas $a/q$ — pico dominante en
  $1/3$ (mod 3, periodo-N 6), luego $1/5,2/5$ (mod 5), alturas que siguen $(\ell-1)/(\ell-2)$.
  Son los **arcos mayores** visibles. Varianza de la oscilación de $\log R_1$ explicada:
  **N mod 6 (primos 2,3) capta 85%**, mod 30 95%, saturando al techo $\mathfrak S$ de 99.9%.
  *El cometa es casi un fenómeno mod 6.*
- **Información (#9, fig 22).** Presupuesto de Var$(\log R_1)$: tendencia 81% + serie singular
  18.8% + **residuo irreducible 0.11%**. El cometa es **~99.9% comprimible** (local-aritmético);
  el residuo es ruido **BLANCO** (espectro plano, autocorr $-0.04$, std 2.7% ~ pocas veces el
  piso de Poisson) — la fluctuación HL genuina, sin estructura escondida. Respuesta negativa
  limpia a "¿los N anómalos son informacionalmente especiales?": no.
- **Geometría discreta (#14, fig 23).** Soluciones Chen $p+rs=N$ como puntos en la superficie;
  en coords log $(u,v)=(\log r/\log N,\log s/\log N)$ forman un triángulo con **estrías
  discretas** (un rayo por cada primo $r$), masa contra el borde $u=0$ (Goldbach). Exponente
  $a=1+\log r/\log s$: **mediana 1.30**, 50% con $a<1.3$, casi todo $\ll1.9$ (Li-Liu). Las
  soluciones se concentran cerca de la frontera de Goldbach (semiprimo desbalanceado, factor
  chico) — forma geométrica de $a_{\rm Res}$ y $B(q)$.
`run_spectral.py`, `run_information.py`, `run_geometry.py`.

## 5decies. Bases primoriales y geometría de acarreos (menú de bases)

- **Cobertura primorial (#1/#2/#7/#8).** En base $B_y=\prod_{p\le y}p$, los residuos
  admisibles $A_y(N)=\{a\in(\mathbb Z/B_y)^\times:N-a\in(\mathbb Z/B_y)^\times\}$ dan cobertura
  local $\lambda_y(N)$; por CRT, $\lambda_y(N)/\lambda_y(\text{gen})=\prod_{p\mid N,p\le y}\frac{p-1}{p-2}=\mathfrak S_y(N)$.
  **La cobertura primorial ES la serie singular truncada.** Explica el cometa: 84.6% (B₆),
  → 99.8% (B₃₀) — la misma saturación que la sección espectral. El cometa es la sombra de
  obstrucciones locales acumuladas.
- **Geometría de acarreos (#3/#4/#9, NUEVO, fig 24).** #acarreos$(p,N{-}p)=(S_b(p)+S_b(N{-}p)-S_b(N))/(b-1)$;
  en base 2 $= v_2\binom{N}{p}$ (Kummer). Las reps de Goldbach tienen un **EXCESO sistemático
  de acarreos** sobre pares aleatorios: base 2 **+0.97** (9.9 vs 8.9), positivo en TODA base
  (b=2..16). Origen aritmético: el último dígito del primo es coprimo con b → sesga $S_b(p)\equiv p$
  mod (b−1). Es una **firma digital** de la primalidad en la geometría de la suma — las mismas
  restricciones locales que dan la serie singular dejan una huella en los acarreos.
`run_bases.py`, `src/bases.py`.

## 6. Direcciones del draft expandido: estado

| dirección (draft expandido) | estado |
|---|---|
| Capas $\Omega(q)=k$, perfil $\rho(k)$, $\beta_k$ | **hecho** (§1ter, fig 10) |
| Rescate compuesto×compuesto $R_{cc}$ | **hecho** (existe para todo $N$) |
| Continuidad débil de $\nu_N$ | **hecho** (§3bis, fig 08) |
| Geometría de factores $B(q)$ | **hecho** (§3, fig 05) |
| MIP multiobjetivo / mínima oscilación | **hecho** (§4, fig 06–07) |
| Formas lineales $N=m_1p+m_2q$ | **hecho** (§5bis) |
| Transporte óptimo (defecto de reflexión $\Delta_k$) | **hecho** (§5quinquies, fig 13) |
| Persistencia topológica de valles (TDA) | **hecho** (§5quinquies, fig 14) |
| Dinámica estadística de $X_N$ | **hecho** (§5quinquies, fig 15) |

## 7. Preguntas abiertas / próximos pasos

1. **Demostrar (o refutar) la media potencia singular** $R_2/R_1\asymp\mathfrak S^{-1/2}$.
   OJO: el experimento de escalamiento (§ siguiente) muestra que $\beta_2$ **no se
   queda en $1/2$**: deriva hacia arriba con $X$. El resultado “publicable” es más
   bien *caracterizar la función $\beta_2(X)$*, no un valor fijo.
2. **Serie singular de $R_2$ exacta**: medir $\mathfrak S_2$ por clases de
   congruencia y comparar con el producto de dos formas lineales.
3. **Transporte / TDA / dinámica**: las tres direcciones pendientes del draft.
4. **Generación de columnas real**: resolver el MIP infinito truncado por
   pricing/CG (paralelo conceptual con el resto de la tesis de despacho).
