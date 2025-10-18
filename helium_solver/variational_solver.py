import numpy as np
from scipy.integrate import nquad

'''
Want toi compute hamiltonian componentents H_ij = <phi_i|H|phi_j>,
and overlap components S_ij = <phi_i|phi_j>.

Total hamltonian for helium (atomix units):
let di = (del_i)^2 (laplacian wrt to electron i)
H = -0.5(d1+d2) - Z/r1 -Z/r2 + 1/r12

Coordiantes r1, r2 in [0, inf)], cos(theta_12) in [-1, 1]
r12 = sqrt(r1^2 +r2^2 - 2r1r2cos(theta_12)) : distance between electron 1 and 2

volume element for these coordinates for integration:

dt1dt2 = 8(pir1r2)^2dr1dr2d(cos(theta_12))

'''

#First will try slater type orbitals as basis: phy_i(r1, r2) = exp(-alpha_i(r1+r2))
def phi(alpha , r1, r2):
    return np.exp(-alpha*(r1+r2))

#Overlap Sij

def r12(r1, r2, costh): #costh := cos(theta_12)
    return np.sqrt(r1**2 + r2**2 - 2*r1*r2*costh)

def integrand_overlap(r1, r2, costh, alpha_i, alpha_j):
    f = phi(alpha_i, r1, r2)
    g = phi(alpha_j, r1, r2)
    return f*g*((r1*r2)**2)

def overlap(alpha_i, alpha_j):
    limits = [ 
        [0, np.inf],
        [0, np.inf],
        [-1, 1],
    ]

    scale = 8*np.pi**2
    return scale*nquad(integrand_overlap, limits, args=(alpha_i, alpha_j), opts={"limit":100})[0]

#Hij (just for potential component V initially):
def integrand_HV(r1, r2, costh, alpha_i, alpha_j, Z=2):
    f = phi(alpha_i, r1, r2)
    g = phi(alpha_j, r1, r2)
    r_12 = r12(r1, r2, costh)
    V = -Z*(1/r2+1/r2)+1/r_12
    return f*V*g*((r1*r2)**2)

def H_potential(alpha_i, alpha_j, Z=2):
    limits = [ 
        [0, np.inf],
        [0, np.inf],
        [-1, 1],
    ]

    scale = 8*np.pi**2
    return scale*nquad(integrand_overlap, limits, args=(alpha_i, alpha_j), opts={"limit":100})[0]
def main():
    i =1
    print(f'overlap({i},{i}) = {overlap(i,i)}')

if __name__ == "__main__":
    main()