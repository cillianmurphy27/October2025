'''
to generate hylleraas type basis functions(curently only radial), will use symmetric 
form so invariant under electron interchange: (r1^m r2^n + r2^m r1^n)r12^p exp(-alpha(r1+r2))

n+m+p <= N_max
'''

import numpy as np
import itertools
#syymetirc
def hy_phi_symmetric(m,n,p, alpha):
    def phi(r1, r2, r12):
        return ((r1**m * r2**n + r2**m *r1**n)*(r12**p)*np.exp(-alpha*(r1+r2)))/np.sqrt(2)
    
    return phi

def hy_phi(m,n,p, alpha):
    def phi(r1, r2, r12):
        return (r1**m * r2**n)*(r12**p)*np.exp(-alpha*(r1+r2))
    
    return phi

#alpha 1.68 close to optimal, will solve for this independantly later
def hylleraas_basis(N_max=2, alpha = 1.68):
    '''
    returns a list of callable Hylleraas basis functions
    '''

    basis = []

    for m,n,p in itertools.product(range(N_max+1), repeat = 3):
        if m+n+p<= N_max:
            if m<n:
                basis.append({
                    'function':hy_phi_symmetric(m,n,p,alpha),
                    'params':(m,n,p,alpha)
                    })
            elif m==n:
                basis.append({
                    'function':hy_phi(m,n,p,alpha),
                    'params':(m,n,p,alpha)
                    })
    
    return basis
    
if __name__ == "__main__":
    print(*hylleraas_basis())