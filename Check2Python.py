import numpy as np
from scipy.stats import norm

def h(x):
    return np.sqrt(x[0, :] ** 2 + x[1, :] ** 2)

def Hh(x):
    x = x.T
    J = np.array([
        x[:, 0] / np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2),
        x[:, 1] / np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2),
    ])
    return J.reshape(1, 2, -1)

# Problem Setup
N     = 4
sigG  = 3
xbark = np.array([-3.5,0])
Pbark = np.array([[1,1/2], [1/2,1]])
y     = 3
R     = np.array([[0.1**2]])
Q     = 2e-1*np.eye(2)
s     = 2
n     = N**2

def gauss(x):
    delta = x - xbark[:, None]
    return np.exp(-0.5 * np.sum(delta * np.linalg.solve(Pbark, delta), axis=0))

Xbark = np.array([
    [-4.59807621135332, -5.59807621135332, -6.59807621135332, -7.59807621135332,
     -2.86602540378444, -3.86602540378444, -4.86602540378444, -5.86602540378444,
     -1.13397459621556, -2.13397459621556, -3.13397459621556, -4.13397459621556,
      0.598076211353315, -0.401923788646684, -1.40192378864668, -2.40192378864668],
    [-4.09807621135332, -3.09807621135332, -2.09807621135332, -1.09807621135332,
     -2.36602540378444, -1.36602540378444, -0.366025403784438,  0.633974596215562,
     -0.633974596215562,  0.366025403784438,  1.36602540378444,  2.36602540378444,
      1.09807621135332,   2.09807621135332,   3.09807621135332,   4.09807621135332]
])

whatkm1  = gauss(Xbark)
whatkm1 /= whatkm1.sum()

# GSF Update
ny     = R.shape[0]  # Assumes R is 1x1 matrix
wbark  = whatkm1
eye_s  = np.eye(s)
Ps     = (4 / (n * (s + 2))) ** (2 / (s + 4)) * Pbark + Q
Ps     = (Ps + Ps.T) / 2
invPs  = np.linalg.inv(Ps)

H      = Hh(Xbark)
Ht     = np.transpose(H,[1,0,2])
HPs    = np.einsum('ikn,kj->ijn',H,Ps)
HPsHt  = np.einsum('ikn,kjn->ijn',HPs,Ht)
W      = HPsHt + R
Winv   = np.moveaxis(np.linalg.inv(np.moveaxis(W,-1,0)),0,-1)
PsHt   = np.einsum('ik,kjn->ijn',Ps,Ht)
K      = np.einsum('ikn,kjn->ijn',PsHt,Winv)
Kt     = np.transpose(K,[1,0,2])
v      = y - h(Xbark)
v      = v.reshape(ny,1,n)
vt     = np.transpose(v,[1,0,2])
Kv     = np.einsum('ikn,kjn->ijn',K,v)
Kv     = Kv.reshape(s,n)
XkGSF  = Xbark + Kv
KH     = np.einsum('ikn,kjn->ijn',K,H)
ImKH   = np.repeat(eye_s,n).reshape(s,s,n) - KH
ImKHt  = np.transpose(ImKH,[1,0,2])
PkGSF  = np.einsum('ikn,kj->ijn',ImKH,Ps)
PkGSF  = np.einsum('ikn,kjn->ijn',PkGSF,ImKHt)
KRK    = np.einsum('ikn,kj->ijn',K,R)
KRK    = np.einsum('ikn,kjn->ijn',KRK,Kt)
PkGSF += KRK
detW   = np.moveaxis(np.linalg.det(np.moveaxis(W,-1,0)),0,-1)
wkGSF  = np.log(wbark) - np.log(np.sqrt(detW.reshape(1,n)))
Wv     = np.einsum('ikn,kjn->ijn',Winv,v)
vtWv   = np.einsum('ikn,kjn->ijn',vt,Wv)
wkGSF += -0.5*vtWv.reshape(1,n)
m      = np.max(wkGSF)
wkGSF  = np.exp(wkGSF - (m + np.log(np.sum(np.exp(wkGSF - m)))))
wkGSF /= np.sum(wkGSF)
xhatk  = XkGSF @ wkGSF.T
Phatk  = np.sum(np.multiply(PkGSF,wkGSF),axis=2)
nuxk   = XkGSF - xhatk
Phatk += nuxk @ np.diag(wkGSF[0]) @ nuxk.T
Phatk  = (Phatk + Phatk.T) / 2