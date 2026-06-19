"""
Independent reproduction of two supplement probes (statistical dynamics 2.4 and the
weak-continuity location law 2.2), from scratch (numpy/FFT only), with NO dependency on src/.

Usage:  python3 reproduce_suppl.py     (X=2e6, ~1 min)
"""
import numpy as np, time
t0 = time.time()
X = 2_000_000
is_prime = np.ones(X+1, dtype=bool); is_prime[:2] = False
for p in range(2, int(X**0.5)+1):
    if is_prime[p]: is_prime[p*p::p] = False
primes = np.nonzero(is_prime)[0]
bigomega = np.zeros(X+1, dtype=np.int16)
for p in primes:
    pe = int(p)
    while pe <= X:
        bigomega[pe::pe] += 1; pe *= p
is_semi = (bigomega == 2)
M = 1
while M < 2*X+1: M <<= 1
def conv(a, b):
    A = np.zeros(M); A[:X+1] = a; B = np.zeros(M); B[:X+1] = b
    return np.rint(np.fft.irfft(np.fft.rfft(A)*np.fft.rfft(B), M)[:X+1]).astype(np.int64)
R1 = conv(is_prime, is_prime); R2 = conv(is_prime, is_semi)
N = np.arange(X+1)
logS = np.zeros(X+1)
for p in primes:
    if p == 2: continue
    logS[p::p] += np.log((p-1)/(p-2))
nz = (R1+R2) > 0
theta = np.where(nz, R1/np.maximum(R1+R2,1), 0.0)

def ols(x, y):
    xm, ym = x.mean(), y.mean()
    b = ((x-xm)*(y-ym)).sum()/((x-xm)**2).sum()
    yhat = ym + b*(x-xm)
    return b, yhat, 1 - ((y-yhat)**2).sum()/((y-ym)**2).sum()

even = (N % 2 == 0)
mask = even & (N>=1000) & (N<=X) & nz
th = theta[mask]; ls = logS[mask]            # ordered by N (step 2)

print("== Suppl 2.4 ==")
# (a) raw regression
_, yh, r2 = ols(ls, th)
res = th - yh
print(f"(a) raw  theta~logS : R^2={r2:.3f} (paper 0.94); resid lag-1 autocorr={np.corrcoef(res[:-1],res[1:])[0,1]:+.2f}")

# (b) local detrend (rolling mean, window 101) then regress
w = 101; k = np.ones(w)/w
def detr(s):
    roll = np.convolve(s, k, 'valid')
    return s[w//2:-(w//2)] - roll
thd, lsd = detr(th), detr(ls)
_, yhd, r2d = ols(lsd, thd)
resd = thd - yhd
print(f"(b) detrended theta~logS : R^2={r2d:.3f}; final resid lag-1 autocorr={np.corrcoef(resd[:-1],resd[1:])[0,1]:+.2f} (paper -0.21)")

# (c) increment vs jump in S
idx = np.nonzero(mask)[0]
idx = idx[idx+2 <= X]
ip2 = idx + 2
g = nz[ip2] & ((ip2 % 2)==0)
dth = theta[ip2[g]] - theta[idx[g]]
dls = logS[ip2[g]] - logS[idx[g]]
_,_,r2i = ols(dls, dth)
print(f"(c) d(theta)~d(logS) : R^2={r2i:.3f} (paper 0.98)")

print("\n== Suppl 2.2  location law (window at 1e6) ==")
lo, Wn = 1_000_000, 4000
qp=[]; qs=[]
for n in range(lo, lo+2*Wn, 2):
    kk = np.searchsorted(primes, n); ps = primes[:kk]; q = n-ps
    qp.append(q[is_prime[q]]/n); qs.append(q[is_semi[q]]/n)
qp=np.concatenate(qp); qs=np.concatenate(qs)
print(f"prime channel  mean q/N = {qp.mean():.3f} (paper 0.500)")
print(f"semiprime chan mean q/N = {qs.mean():.3f} (paper 0.513)")
B=50; hp,_=np.histogram(qp,B,(0,1)); hs,_=np.histogram(qs,B,(0,1))
hp=hp/hp.sum(); hs=hs/hs.sum(); u=np.ones(B)/B
print(f"TV(prime, uniform)   = {0.5*np.abs(hp-u).sum():.3f} (paper 0.028)")
print(f"TV(prime, semiprime) = {0.5*np.abs(hp-hs).sum():.3f} (paper 0.019)")
print(f"\nelapsed {time.time()-t0:.1f}s")
