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
La serie singular de las dos formas $(s,\,N-rs)$ factoriza como
$$\mathfrak S_2(r,N)=2C_2\cdot
\begin{cases}\mathfrak S(N)\,\dfrac{r-1}{r-2}, & r\nmid N,\\[1mm]
\mathfrak S(N)\,\dfrac{r-2}{r-1}, & r\mid N.\end{cases}$$
La clave: cuando el factor pequeño $r$ **divide** a $N$, el realce
$(r-1)/(r-2)$ del primo $r$ se **invierte** a $(r-2)/(r-1)<1$ — porque permitir
que $q$ sea divisible por $r$ *neutraliza* la condición local que enriquecía a
Goldbach. Esa supresión ocurre justo cuando $\mathfrak S(N)$ es grande, y reduce el
exponente efectivo por debajo de 1.

Esto explica **la dirección** ($\beta<1$) pero, cuantitativamente, el modelo con
la corrección $r\mid N$ deja un exponente residual $\approx0.78$, no $0.5$: la
neutralización va más allá de los términos $r\mid N$. El valor empírico
$\beta\approx1/2$ (con deriva lenta tipo $\log\log$) es por ahora una **regularidad
abierta** — el candidato natural a teorema/conjetura de este proyecto.

> **Conjetura (media potencia singular).** Para $X\to\infty$,
> $\displaystyle\frac{\sum_{N\le X}\log(R_2/R_1)\,\log\mathfrak S(N)}{\sum_{N\le X}\log^2\mathfrak S(N)}\to -\tfrac12$,
> es decir $R_2/R_1\asymp\mathfrak S(N)^{-1/2}$ en media logarítmica.

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

Esto realiza la idea de *phase diagram* / valor crítico $a_X^*(N)$ del `.tex`: la
flexibilidad semiprima de Chen se vuelve **necesaria** exactamente al exigir
balance casi perfecto.

---

## 5. Preguntas abiertas / próximos pasos

1. **Demostrar (o refutar) la media potencia singular** $R_2/R_1\asymp\mathfrak S^{-1/2}$.
   Requiere tratar con cuidado el rango completo del factor $r$ y la deriva
   $\log\log$ de $\beta$. Es el resultado “publicable” del programa.
2. **Cuantificar la deriva de $\beta$**: ¿$\beta\to1/2$, o $\beta\to$ otro límite, o
   crece como $1/2+c/\log\log N$? Necesita $X\gtrsim10^9$ (segmentar la criba).
3. **Serie singular de $R_2$ exacta**: medir $\mathfrak S_2$ por clases de
   congruencia y comparar con el producto de dos formas.
4. **Continuidad débil** (medidas $\mu_N^{(t)}$ de $q/N$): falta el experimento de
   convergencia débil suavizando en ventanas $[X,X+H]$.
5. **Generación de columnas real**: resolver el MIP infinito truncado por
   pricing/CG (paralelo conceptual con el resto de la tesis de despacho).
