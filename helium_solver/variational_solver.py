import numpy as np
from scipy.integrate import nquad
from scipy.linalg import eigh

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
def STO_phi(alpha , r1, r2):
    normalization = (alpha**3)/np.pi # normalization factor for basis phi
    return normalization*np.exp(-alpha*(r1+r2))

#Overlap Sij
def r12(r1, r2, costh): #costh := cos(theta_12)
    return np.sqrt(r1**2 + r2**2 - 2*r1*r2*costh)

def STO_integrand_overlap(r1, r2, costh, alpha_i, alpha_j):
    f = STO_phi(alpha_i, r1, r2)
    g = STO_phi(alpha_j, r1, r2)
    return f*g*((r1*r2)**2)

def Hy_integrand_overlap(r1, r2, costh, phi_i, phi_j):
    r_12 = r12(r1,r2, costh)
    return phi_i(r1,r2,costh)*phi_j(r1,r2,costh)*((r1*r2)**2)


def STO_overlap(alpha_i, alpha_j):
    limits = [ 
        [0, np.inf],
        [0, np.inf],
        [-1, 1],
    ]

    scale = 8*np.pi**2
    return scale*nquad(STO_integrand_overlap, limits, args=(alpha_i, alpha_j), opts={"limit":100})[0]

def Hy_overlap(phi_i, phi_j):
    limits = [ 
        [0, np.inf],
        [0, np.inf],
        [-1, 1],
    ]
    scale = 8*np.pi**2
    return scale*nquad(Hy_integrand_overlap, limits, args=(phi_i, phi_j), opts={"limit":100})[0]

#Hij (just for potential component V initially):
def STO_integrand_HV(r1, r2, costh, alpha_i, alpha_j, Z=2):
    f = STO_phi(alpha_i, r1, r2)
    g = STO_phi(alpha_j, r1, r2)
    r_12 = r12(r1, r2, costh)
    V = -Z*(1/r2+1/r2)+1/r_12
    return f*V*g*((r1*r2)**2)

def Hy_integrand_HV(r1, r2, costh, phi_i, phi_j, Z=2):
    r_12 = r12(r1, r2, costh)
    V = -Z*(1/r2+1/r2)+1/r_12
    return phi_i(r1,r2,r_12)*V*phi_j(r1,r2,r_12)*((r1*r2)**2)

def STO_H_potential(alpha_i, alpha_j, Z=2):
    limits = [ 
        [0, np.inf],
        [0, np.inf],
        [-1, 1],
    ]

    scale = 8*np.pi**2
    return scale*nquad(STO_integrand_HV, limits, args=(alpha_i, alpha_j), opts={"limit":100})[0]

def Hy_H_potential(phi_i, phi_j):
    limits = [ 
        [0, np.inf],
        [0, np.inf],
        [-1, 1],
    ]
    scale = 8*np.pi**2
    return scale*nquad(Hy_integrand_HV, limits, args=(phi_i, phi_j), opts={"limit":100})[0]


#numerical integration from -1 to 1 integral_a_^b f(x) ~ Sum_i^n wi *f(xi)
from scipy.special import roots_legendre, roots_laguerre
def Gauss_Legendre(f, a =-1 , b=1, n=100):
    x, w = roots_legendre(n)
    t = 0.5*(b-a)*x + 0.5*(b+a)
    return 0.5*(b-a) * np.sum(w*f(t))

#numerical integration for intgeral_0^inf exp(-x)f(x) ~ sum_i^n wi * f(xi)
def Gauss_Laguerre(f, n =100):
    x, w = roots_laguerre(n)
    return np.sum(w * f(x))

#analytic result for kinetic term of H with STO's
#TODO: develop numerical method to solve kinetic term for hylleraas basis fumctions
'''
as only using radial functions for basis laplacian del^2 g(r) = g''(r) + 2/r * g'(r)
For STO's and hamiltonian term above applied to phi_j this reduces to: -(alpha_j^2 - alpha_j(1/r1 + 1/r2))phi_j

'''
def STO_H_kinetic(alpha_i, alpha_j, Nr=40, Nu =40):
 #number for sum for radial components Nr
 #number of sum for cos(theta) components Nu

    factor = 8*((np.pi)**2)
    beta = alpha_i +alpha_j

    # Gauss-Laguerre nodes/weights for integral int_0^inf e^{-x} f(x) dx
    x_nodes, x_w = roots_laguerre(Nr)
    # Gauss-Legendre nodes/weights for mu in [-1,1]
    u_nodes, u_w = roots_legendre(Nu)

    normalization = ((alpha_i**3)/np.pi)*((alpha_j**3)/np.pi)

    T = 0 #initialize sum to zero
    #Triple integral(sum) 
    for p in range(Nr):
        x_p = x_nodes[p]
        w_p = x_w[p]
        #change of variables from r -> xp/beta: dr -> dxp/beta
        r1 = x_p/beta
        jac_r1 = 1.0/beta

        for q in range(Nr):
            x_q = x_nodes[q]
            w_q = x_w[q]
            r2 = x_q/beta
            jac_r2 = 1.0/beta

            laplacian = -(alpha_j**2 - alpha_j*(1/r1 + 1/r2))

            #this is f(x) in int_0^inf exp(-x)*f(x)
            F_r = normalization*laplacian*r1*r1*r2*r2
            #phi_ij = phi(alpha_i,r1, r2)*phi(alpha_j, r1, r2)
            radial_weight = w_p*w_q*jac_r1*jac_r2

            #Don't need angular integral for STO's but will for more complex basis functions
            for k in range(Nu):
                u= u_nodes[k]
                w_u = u_w[k]

                
                #full intgrand including jacobian r1^2 r2^2
                integrand = F_r *radial_weight*w_u

                #accumulate with weights: integrand*radial_weight*w_u
                T += integrand

    return factor*T
    #analytic result
    #return 64 * (alpha_i *alpha_j)**4 / ((alpha_i +alpha_j)**6) 

'''
let a := alpha 
Exact form of (del_1^2 +del_2^2)phi_i = added to latex file which must be added to git
'''

def Hy_lapacian(phi_i):
    params = phi_i['params']
    m,n,p,a = params[0], params[1], params[2], params[3]
    #This is a fucking mess, might need to try sympy for symbolic differentiation
    def laplacian(r1, r2, r12):
        t1 = (2*a**2)*(r1**m * r2**n * r12**p)*(r1/r2 +1 + r2/r2)
        t2 = -2*a*(r1**(m-1) * r2**(n-1) * r12**(p-1))*(m*r2*r12 + n*r1*r12 + 2*p*r1*r2) 
        t3 = (r1**(m-2))*(r2**(n-2))*(r12**(p-2))*(m*(m-1)*r2**2 * r12**2 + n*(n-1)*r1**2*r12**2 + 2*p*(p-1)*r1**2*r2**2)
        t4 = (2*r1**(m-1)*r2**(n-1)*r12**(p-1))*((m*r2*r12)/r1 + (n*r1*r12)/r2+ (2*p*r1*r2)/r12)
        t5 = 2*((r1**2 + r2**2 - r12**2)/(r1*r2))*(m*n*r1**(m-1)*r1**(n-1)*r12**p - a*m*r1**(m)*r2**(n-1)*r12**p - a*n*r1**(m-1)*r2**n*r12**p + a**2*r1**m*r2**n*r12**p)
        t6 = 2*((r1**2 - r2**2 + r12**2)/(r1*r12))*(m*p*r1**(m-1)*r2**n*r12**(p-1) - a*p*r1**m*r2**n*r12*(p-1))
        t7 = 2*((-r1**2 + r2**2 + r12**2)/(r2*r12))*(n*p*r1**(m)*r2**(n-1)*r12**(p-1) - a*p*r1**m*r2**n*r12*(p-1))
        P= t1+t2+t3+t4+t5+t6+t7
        
        return P 
    return laplacian


def initialize_matrices(alphas, hylleraas=False):
    n = len(alphas)

    #Initialize H and S matrices as 0,0s
    H = np.zeros((n,n))
    S = np.zeros((n,n))

    #Populate matrices with values from integrals
    if hylleraas:
        pass
    else:
        for i in range(n):
            for j in range(n):
                S[i,j] = STO_overlap(alphas[i], alphas[j])
                H[i,j] = STO_H_potential(alphas[i], alphas[j]) + STO_H_kinetic(alphas[i], alphas[j])
    
    return H, S

def main():
    alphas = [ 0.5, 1, 1.5] #alphas for basis functions
    H, S = initialize_matrices(alphas)
    E, C = eigh(H, S)
    print(f"alpha values = {alphas}")
    print(f"Energy eigenvalues: {E}")
    print(f"Eigenvectors C = {C}")

if __name__ == "__main__":
    main()