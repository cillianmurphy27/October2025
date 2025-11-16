from skopt import gp_minimize
from variational_solver import solve

def objective_function(alphas):
    '''Objective function for Bayesian optimization'''
    #alpha = params[0]

    energy, _ = solve(alphas)
    return energy

def main():
    bounds = [(1.0, 3.0)]*7

    res = gp_minimize(
        objective_function,
        bounds,
        n_calls=200,
        random_state=42,
        verbose = True 
    )

    print("Optimal alpha vector (α₁...α₇):")
    print(f"{[f'{a:.4f}' for a in res.x]}")
    print(f"Minimum energy: {res.fun:.8f} a.u.")

    return 0

if __name__ == "__main__":
    main()

'''
bounds = [(1.0, 2.6)]

res = gp_minimize(
    objective_function,
    bounds,
    n_calls=20,
    random_state=42
)

Optimal alpha: 1.8534
Minimum energy: -2.90142753 a.u.

Mulit-alpha:
N_max =2
Optimal alpha vector (α₁...α₇):
['2.4151', '1.6208', '2.4938', '2.4152', '1.6661', '2.1889', '1.9190']
Minimum energy: -2.90149051 a.u.
'''