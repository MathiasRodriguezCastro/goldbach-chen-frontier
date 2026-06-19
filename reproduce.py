"""
Independent reproduction of the empirical core, from scratch (numpy/FFT only), with NO
dependency on src/ or the rest of the repository. Confirms, against a separate implementation:
pi(X), the prime/semiprime counts, Goldbach (min R1>0), the R1 singular-series control
(slope 1.000, CV collapse on dividing by S), the cumulative beta_2 drift (0.45->0.48->0.50),
the layer sign reversal (beta_k crosses zero between k=2 and k=3), and theta/Chen-surplus.

Usage:  python3 reproduce.py     (X=2e6, ~1 min)
"""
import numpy as np, time
t0 = time.time()
X = 2_000_000

# ---- prime sieve ----
is_prime = np.ones(X+1, dtype=bool); is_prime[:2] = False
for p in range(2, int(X**0.5)+1):
    if is_prime[p]: is_prime[p*p::p] = False
primes = np.nonzero(is_prime)[0]
print(f"pi(X)            = {len(primes)}")

# ---- Omega(n) via prime-power sieve ----
bigomega = np.zeros(X+1, dtype=np.int16)
for p in primes:
    pe = int(p)
    while pe <= X:
        bigomega[pe::pe] += 1
        pe *= p
is_semi = (bigomega == 2)
print(f"#semiprimes <= X = {int(is_semi.sum())}")

# ---- FFT convolutions: R1 = 1P*1P, R2 = 1P*1S2 ----
M = 1
while M < 2*X+1: M <<= 1
def conv(a, b):
    A = np.zeros(M); A[:X+1] = a
    B = np.zeros(M); B[:X+1] = b
    return np.rint(np.fft.irfft(np.fft.rfft(A)*np.fft.rfft(B), M)[:X+1]).astype(np.int64)
R1 = conv(is_prime, is_prime)
R2 = conv(is_prime, is_semi)

# ---- brute-force validation (vectorized) ----
rng = np.random.default_rng(0)
sample = rng.choice(np.arange(4, X+1, 2), size=10, replace=False)
ok = True
for n in sample:
    a = is_prime[:n+1]
    b1 = int((a & a[::-1]).sum())
    b2 = int((a & is_semi[:n+1][::-1]).sum())
    if b1 != R1[n] or b2 != R2[n]:
        ok = False; print(f"  MISMATCH N={n}")
print(f"brute-force check (10 random N): {'PASS' if ok else 'FAIL'}")
print(f"min R1 over even N>=4 (Goldbach): {int(R1[4:X+1:2].min())}")

# ---- singular series logS = sum_{odd p|N} log((p-1)/(p-2)) ----
N = np.arange(X+1)
logS = np.zeros(X+1)
for p in primes:
    if p == 2: continue
    logS[p::p] += np.log((p-1)/(p-2))
S = np.exp(logS)
twoC2 = 2*0.6601618158

def ols(x, y):
    xm, ym = x.mean(), y.mean()
    b = ((x-xm)*(y-ym)).sum()/((x-xm)**2).sum()
    yhat = ym + b*(x-xm)
    r2 = 1 - ((y-yhat)**2).sum()/((y-ym)**2).sum()
    return b, r2

logN = np.log(np.maximum(N, 2)); norm = N/logN**2
even = (N % 2 == 0)

# ---- R1 control: slope on log S should be ~1.000 ----
m1 = even & (N>=5000) & (N<=X) & (R1>0)
b1, r2_1 = ols(logS[m1], np.log(R1[m1]/norm[m1]))
print(f"\n[R1 control] slope of log(R1/(N/log^2N)) on log S = {b1:.4f}  (R^2={r2_1:.4f})")
nc  = R1[m1]/norm[m1]
ncS = R1[m1]/(S[m1]*norm[m1])
print(f"[R1 CV] before /S = {nc.std()/nc.mean():.3f} ; after /S = {ncS.std()/ncS.mean():.4f} ; "
      f"reduction = {(nc.std()/nc.mean())/(ncS.std()/ncS.mean()):.1f}x")
print(f"[R1 const] mean R1/(2C2*S*N/log^2N) = {(R1[m1]/(twoC2*S[m1]*norm[m1])).mean():.3f}")

# ---- beta2 = 1 + slope[log(R2/R1) on log S], cumulative over [5000,Xtop] ----
print()
for Xtop in [500_000, 1_000_000, 2_000_000]:
    m = even & (N>=5000) & (N<=Xtop) & (R1>0) & (R2>0)
    b, r2 = ols(logS[m], np.log(R2[m]/R1[m]))
    print(f"[beta2] X={Xtop:>9}: beta2 = {1+b:.3f}  (R^2={r2:.3f}, n={int(m.sum())})")

# ---- layers R_k and exponents (sign reversal) ----
print("\n[layers] k : share% : beta_k")
FP = np.fft.rfft(np.concatenate([is_prime.astype(float), np.zeros(M-(X+1))]))
def convP(ind):
    B = np.zeros(M); B[:X+1] = ind
    return np.rint(np.fft.irfft(FP*np.fft.rfft(B), M)[:X+1]).astype(np.int64)
mr = even & (N>=5000) & (N<=X)
Rk = {k: convP(bigomega==k) for k in range(1,8)}
tot = sum(Rk[k][mr].sum() for k in range(1,8))
for k in range(1,8):
    share = Rk[k][mr].sum()/tot*100
    if k == 1:
        print(f"  k={k} : {share:5.1f}% : +1.00 (reference)"); continue
    mk = mr & (R1>0) & (Rk[k]>0)
    bk,_ = ols(logS[mk], np.log(Rk[k][mk]/R1[mk]))
    print(f"  k={k} : {share:5.1f}% : {1+bk:+.2f}")

# ---- a couple of cheap diagnostics ----
md = even & (N>=1000) & (N<=X) & (R1>0)
theta = (R1[md]/(R1[md]+R2[md])).mean()
csurp = (R2[md]/(R1[md]+1)).mean()
print(f"\n[diag] mean prime share theta = {theta:.3f} ;  mean Chen surplus C = {csurp:.2f}")
print(f"elapsed {time.time()-t0:.1f}s")
