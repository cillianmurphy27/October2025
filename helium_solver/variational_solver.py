import numpy as np
from scipy.integrate import nquad
from scipy.linalg import eigh
from scipy.special import roots_legendre, roots_laguerre
import itertools

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

#modified form hyllerass.py file onlyreturn the params 
#alpha 1.68 close to optimal, will solve for this independantly later
def hylleraas_basis(N_max=2, alpha = 1.68):
    '''
    Returns a list of Hylleraas basis parameters (m,n,p,alpha).
    Generates unique, symmetric combinations where m+n+p <= N_max.
    '''

    basis = []

    for m,n,p in itertools.product(range(N_max+1), repeat = 3):
        # Only include terms where m >= n to avoid duplicates
        # e.g., (m=0, n=1) is the same as (m=1, n=0)
        # We also enforce the N_max condition
        if m+n+p <= N_max and m >= n:
            basis.append((m, n, p, alpha))
    
    return basis

def r12(r1, r2, costh): #costh := cos(theta_12)
    #for stabilty
    costh = np.clip(costh, -1.0, 1.0)
    return np.sqrt(r1**2 + r2**2 - 2*r1*r2*costh)

def phi_polynomial(m, n, p, r1, r2, r_12):
    '''
    Calculates teh polynomial part of Hyleraas basis function.
    '''
    if m == n:
        return (r1**m *r2**n)*(r_12**p)
    else:
        return ((r1**m*r2**n + r2**m*r1**n)*(r_12**p))/np.sqrt(2)
    
def phi_derivatives(m,n,p,r1, r2, r_12):
    '''
    Calculates the derivatives of teh polynomial part of phi wrt r1, r2, r12
    '''

    #To avoid divison by 0
    if r1==0: r1 = 1e-10
    if r2==0: r2 = 1e-10
    if r_12==0: r_12 = 1e-10

    #exp_factor = np.exp(-alpha*(r1+r2))

    #polynomial part of derivative
    if m==n:
        f = (r1**m)*r2**n
        df_dr1 = m*(r1**(m-1))*r2**n
        df_dr2 = n*(r1**m)*(r2**(n-1))

    else:
        f = (r1**m*r2**n + r2**m*r1**n)/np.sqrt(2)
        df_dr1 = (m*(r1**(m-1))*r2**n + n*r2**m*(r1**(n-1)))/np.sqrt(2)
        df_dr2 = (n*(r1**m)*(r2**(n-1)) + m*(r2**(m-1))*(r1**n))/np.sqrt(2)

    g = r_12**p
    dg_dr12 = p*r_12**(p-1)

    polynomial = f*g
    dphi_dr1 = df_dr1 *g
    dphi_dr2 = df_dr2 *g
    dphi_dr12 = f *dg_dr12

    '''
    #include exponential part in the derivative using product rule
    dphi_dr1 = (dphi_dr1 - alpha*polynomial)*exp_factor
    dphi_dr2 = (dphi_dr2 - alpha*polynomial)*exp_factor
    dphi_dr12 = dphi_dr12*exp_factor
    '''
    

    return dphi_dr1, dphi_dr2, dphi_dr12

def integrand_S(r1, r2, r_12, mu, params_i, params_j, Z=2):
    m_i, n_i, p_i, _ = params_i
    m_j, n_j, p_j, _ = params_j

    phi_i = phi_polynomial(m_i, n_i, p_i, r1, r2, r_12)
    phi_j = phi_polynomial(m_j, n_j, p_j, r1, r2, r_12)
    return phi_i*phi_j

def integrand_V(r1, r2, r_12, mu, params_i, params_j, Z=2):
    m_i, n_i, p_i, _ = params_i
    m_j, n_j, p_j, _ = params_j

    phi_i = phi_polynomial(m_i, n_i, p_i, r1, r2, r_12)
    phi_j = phi_polynomial(m_j, n_j, p_j, r1, r2, r_12)

    V = -Z*(1/r1 + 1/r2) + 1/r_12
    return phi_i *V *phi_j

def integrand_T(r1, r2, r_12, mu, params_i, params_j, Z=2):
    '''
    m_i, n_i, p_i, alpha_i = params_i
    m_j, n_j, p_j, alpha_j = params_j

    di_dr1, di_dr2, di_dr12 = phi_derivatives(m_i, n_i, p_i, alpha_i, r1, r2, r_12)
    dj_dr1, dj_dr2, dj_dr12 = phi_derivatives(m_j, n_j, p_j, alpha_j, r1, r2, r_12)

    term1 = di_dr1 * dj_dr1 + di_dr12 * dj_dr12 + \
            ((r1**2 - r2**2 + r_12**2) / (2 * r1 * r_12)) * (di_dr1 * dj_dr12 + di_dr12 * dj_dr1)
    term2 = di_dr2 * dj_dr2 + di_dr12 * dj_dr12 + \
            ((r2**2 - r1**2 + r_12**2) / (2 * r2 * r_12)) * (di_dr2 * dj_dr12 + di_dr12 * dj_dr2)
    '''
    # Get alpha values
    a_i = params_i[3]
    a_j = params_j[3]

    # Get polynomial values P_i and P_j
    m_i, n_i, p_i, _ = params_i
    m_j, n_j, p_j, _ = params_j
    P_i = phi_polynomial(m_i, n_i, p_i, r1, r2, r_12)
    P_j = phi_polynomial(m_j, n_j, p_j, r1, r2, r_12)

    # Get polynomial derivatives for P_i: (dPi/dr1, dPi/dr2, dPi/dr12)
    di_dr1, di_dr2, di_dr12 = phi_derivatives(m_i, n_i, p_i, r1, r2, r_12)
    # Get polynomial derivatives for P_j: (dPj/dr1, dPj/dr2, dPj/dr12)
    dj_dr1, dj_dr2, dj_dr12 = phi_derivatives(m_j, n_j, p_j, r1, r2, r_12)

    # --- Define dot product terms for clarity ---
    # These are the terms my previous function simplified (incorrectly).

    # Dot product (grad_1 . r1_hat)
    r1_dot_r12_hat = (r1**2 - r2**2 + r_12**2) / (2 * r1 * r_12)
    grad1_Pi_dot_r1_hat = di_dr1 + r1_dot_r12_hat * di_dr12
    grad1_Pj_dot_r1_hat = dj_dr1 + r1_dot_r12_hat * dj_dr12

    # Dot product (grad_2 . r2_hat)
    r2_dot_r12_hat = (r2**2 - r1**2 + r_12**2) / (2 * r2 * r_12)
    grad2_Pi_dot_r2_hat = di_dr2 + r2_dot_r12_hat * di_dr12
    grad2_Pj_dot_r2_hat = dj_dr2 + r2_dot_r12_hat * dj_dr12
    
    # --- Term 1: (nabla_1 phi_i . nabla_1 phi_j) ---

    # (grad_1 P_i . grad_1 P_j) in Hylleraas coordinates
    grad1_Pi_dot_grad1_Pj = di_dr1 * dj_dr1 + di_dr12 * dj_dr12 + \
        r1_dot_r12_hat * (di_dr1 * dj_dr12 + di_dr12 * dj_dr1)

    # This is the full (nabla_1 phi_i . nabla_1 phi_j) / (Ei*Ej)
    term1_full = (
        grad1_Pi_dot_grad1_Pj       # Term 1: (grad_1 P_i . grad_1 P_j)
        - a_j * P_j * grad1_Pi_dot_r1_hat  # Term 2: -a_j * (grad_1 P_i . r1_hat) * P_j
        - a_i * P_i * grad1_Pj_dot_r1_hat  # Term 3: -a_i * (r1_hat . grad_1 P_j) * P_i
        + a_i * a_j * P_i * P_j       # Term 4: +a_i*a_j * (r1_hat . r1_hat) * P_i*P_j
    )

    # --- Term 2: (nabla_2 phi_i . nabla_2 phi_j) ---

    # (grad_2 P_i . grad_2 P_j) in Hylleraas coordinates
    grad2_Pi_dot_grad2_Pj = di_dr2 * dj_dr2 + di_dr12 * dj_dr12 + \
        r2_dot_r12_hat * (di_dr2 * dj_dr12 + di_dr12 * dj_dr2)
        
    # This is the full (nabla_2 phi_i . nabla_2 phi_j) / (Ei*Ej)
    term2_full = (
        grad2_Pi_dot_grad2_Pj       # Term 1
        - a_j * P_j * grad2_Pi_dot_r2_hat  # Term 2
        - a_i * P_i * grad2_Pj_dot_r2_hat  # Term 3
        + a_i * a_j * P_i * P_j       # Term 4
    )
    
    # The total kinetic integrand is 0.5 * (term1_full + term2_full)
    return 0.5 * (term1_full + term2_full)

def compute_integral(integrand, params_i, params_j, Z, Nr=40, Nu=40):
    beta = params_i[3] + params_j[3]
    x_nodes, x_w = roots_laguerre(Nr)
    u_nodes, u_w = roots_legendre(Nu)
    T = 0
    for p in range(Nr):
        x_p = x_nodes[p]
        w_p = x_w[p]
        r1 = x_p/beta
        jac_r1 = 1.0/beta
        for q in range(Nr):
            x_q = x_nodes[q]
            w_q = x_w[q]
            r2 = x_q/beta
            jac_r2 = 1.0/beta

            F_r_mu = 0
            for k in range(Nu):
                mu = u_nodes[k]
                w_mu = u_w[k]
                r_12 = r12(r1, r2, mu)
                if r_12 < 1e-10:
                    continue
                integrand_val = integrand(r1, r2, r_12, mu, params_i, params_j, Z)
                F_r_mu += w_mu*integrand_val
            total_integrand = F_r_mu * (r1**2) * (r2**2)
            T += w_p*w_q *total_integrand *jac_r1*jac_r2
    return 8*(np.pi**2)*T 

def initialize_matrices(basis, Z=2):
    n = len(basis)

    #Initialize H and S matrices as 0,0s
    H = np.zeros((n,n))
    S = np.zeros((n,n))

    #Populate matrices with values from integrals
    print(f"\n ---Building S and H matirces for {n}x{n} basis ---")
    for i in range(n):
            for j in range(i, n):
                params_i = basis[i]
                params_j = basis[j]
                S_ij = compute_integral(integrand_S, params_i, params_j, Z)
                V_ij = compute_integral(integrand_V, params_i, params_j, Z)
                T_ij = compute_integral(integrand_T, params_i, params_j, Z)
                H_ij = V_ij + T_ij
                S[i,j] = S[j, i] = S_ij
                H[i,j] = H[j, i] = H_ij
    
    return H, S

GROUND_STATE_E = -2.90372
def main():
    N_max = 4
    alpha = 1.68
    Z = 2

    basis = hylleraas_basis(N_max, alpha)
    H, S = initialize_matrices(basis, Z)
    E, C = eigh(H, S)

    print("\n ---Results---")
    print(f"Basis set size (N_max = {N_max}): {len(basis)} functions")
    print(f"Non-Linear paramter (alpha): {alpha}")
    print(f"Calculated Ground State Energy: {E[0]:.8f} a.u.")
    print(f"Actual Ground State Energy = {GROUND_STATE_E} a.u. ")
    assert E[0] >= GROUND_STATE_E

if __name__ == "__main__":
    main()
'''
Results:
N=4, 22x22 matirces E_0 = -2.90153083
N=3, 13x13 matrices E_0 = -2.90139411
N=2, 7x7 matrices E_0 = -2.90092400
N=1, 3x3 matrices E_0 = -2.88772343
'''