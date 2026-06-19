# Cover letter

**To:** The Editors-in-Chief, *Experimental Mathematics*
**Re:** Submission of *The Goldbach–Chen Frontier: A Singular-Series Cancellation behind a Finite-Range Half-Power*
**Author:** Mathias Rodríguez Castro (ORCID 0009-0002-7235-7677)

<!-- Header note: "Dear Editors" is the safe salutation. If you confirm the current
Editor-in-Chief on the EM masthead (S. Tabachnikov has held the post for years), you may
address them by name instead. -->

Dear Editors,

I am pleased to submit the manuscript above for consideration in *Experimental Mathematics*.

**The experimental finding.** Separating the Goldbach count $R_1(N)$ (prime $q$) from the
Chen-surplus count $R_2(N)$ (semiprime $q$), a finite-range log–log regression of $R_2/R_1$
against the Hardy–Littlewood singular series $\mathfrak S(N)$ assigns the semiprime channel an
apparent exponent $\beta_2\approx\tfrac12$, strikingly unlike the $\beta_1=1$ of the prime
channel. Computing $R_1,R_2$ to $X=10^9$ by a block-convolution sieve shows this is **not** a
structural half-power: the effective exponent drifts upward, a drift a moving-block bootstrap
makes statistically decisive.

**The object, and the explanation.** The contribution is an object, not a local computation: the
*channel ratio* $R_2/R_1$ at fixed $N$, in which $\mathfrak S(N)$ cancels exactly. Decomposing a
semiprime $q=rs$ by its smaller prime factor, each channel inherits the *same* $\mathfrak S(N)$
that governs $R_1$ (an elementary Euler-product identity), so $\mathfrak S$ enters $R_2$ with
exponent $1$ and cancels in the ratio, leaving a Mertens channel sum $W_f(N)\sim\log\log N$. This
is distinct from the shifted-correlation literature (primes shifted to almost-primes), which fixes
a gap and averages over the summand — there the two singular series do *not* cancel. One identity
then organizes the whole picture: the channel ratio, the sign reversal of the effective exponent
across arithmetic-complexity layers, the residual collapse, and the Chen-rescue threshold.

**What is proved, and what is conjectured.** I am explicit about status. Unconditional are (i) the
cancellation identity and (ii) a linear almost-all asymptotic $R_2(N)=\mathrm{main}(N)(1+o(1))$ for
all but $o(X)$ even $N$, obtained via a circle-method second moment together with the
equidistribution of semiprimes in arithmetic progressions. Upgrading this to the log-regression
statement $\beta_2(N)\to1$ requires one further input, isolated as an explicit, unproven
lower-tail hypothesis on $R_2/R_1$; the functional form $1-\beta_2\asymp1/\log\log N$ is proved in
a Cramér model and, for the primes, conditionally on a quantitative Hardy–Littlewood input. A
"Status of the results" table and an "Open problems and conjectures" section state precisely what
is proved, what is conditional, and what is conjectured.

**Fit with the journal.** The manuscript falls within the journal's stated scope in three of its
own terms: a *formal result inspired by experimentation* — the unconditional almost-all asymptotic
for the semiprime channel, prompted by the regression; *experimental data supporting a significant
hypothesis* — the $\beta_2\to1$ drift computed to $10^9$, with a block bootstrap and a
singular-series-matched Cramér trajectory; and a *conjecture suggested by that data* — the
lower-tail hypothesis, stated in full. It is not a proof of Goldbach or of Chen, and does not
claim to be.

**Reproducibility, and an external check.** Every figure, table, and quoted number is regenerated
by an open-source repository (`run_all.py`; pinned `requirements.txt`; per-module self-tests),
public on GitHub and archived on Zenodo (doi:10.5281/zenodo.20725701). In addition, the empirical
core was *independently reproduced* by a separate from-scratch implementation (`reproduce.py`,
numpy/FFT only, no shared code), recovering the prime/semiprime counts, the $R_1$ singular-series
control, the $\beta_2$ drift, and the layer sign reversal to the quoted precision.

The manuscript is single-author and has not been submitted elsewhere. I would be glad to provide
any further material.

Sincerely,
Mathias Rodríguez Castro
