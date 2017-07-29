import numpy as np
'''
def legendre_gauss(n):
    tau, w = np.polynomial.legendre.leggauss(n)
    tau = np.hstack((-1,tau,1))
    return tau, w

#in: LG-nodes
#out: Gaussian Derivative Matrix#

def deriv_mat(tau):
    tau = np.sort(tau)
    n = tau.shape[0]-2
    D = np.zeros([n,n+1])
    for k in range(1,n+1):
        for l in range(0,n+1):
            if k==l:
                D[k-1,l] = 0
                for m in range(0,n+1):
                    if m != k:
                        D[k-1,l] += 1.0/(tau[k]-tau[m])
            else:
                D[k-1,l] = 1.0/(tau[l]-tau[k])
                for m in range(0,n+1):
                    if m != k and m != l:
                        D[k-1,l] *= (tau[k]-tau[m])/(tau[l]-tau[m])
    return D
'''
def make_node_derivative_matrix(n):
    beta = np.array([0.5 / np.sqrt(1-(2*i)**(-2)) for i in range(1,n)])
    T = np.diag(beta,1) + np.diag(beta,-1)
    D_, V = np.linalg.eig(T)
    tau = np.sort(D_)
    i = np.argsort(D_)
    w = 2 * (V[0,i]**2)
    tau = np.hstack((-1,tau,1))
    D = np.zeros([n,n+1])
    for k in range(1,n+1):
        for l in range(0,n+1):
            if k==l:
                D[k-1,l] = 0
                for m in range(0,n+1):
                    if m != k:
                        D[k-1,l] += 1.0/(tau[k]-tau[m])
            else:
                D[k-1,l] = 1.0/(tau[l]-tau[k])
                for m in range(0,n+1):
                    if m != k and m != l:
                        D[k-1,l] *= (tau[k]-tau[m])/(tau[l]-tau[m])
    return tau, w, D