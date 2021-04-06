import numpy as np
from scipy import optimize

def u_func(c, h, mp):
 
    return (c**(1-mp['phi']))*(h**mp['phi'])

def tau(h, mp, p = 1):


    p_tilde = p*h*mp['epsilon']
    return mp['tau_g']*p_tilde + mp['tau_p']*(max(p_tilde - mp['p_bar'], 0))

def user_cost(h, mp, p=1):

    taxes = tau(h, mp, p)
    interest = mp['r']*h*p

    return interest + taxes

def choose_c(h, m, mp, p=1):


    return m - user_cost(h, mp, p)

def value_of_choice(h, m, mp, p=1):


    c = choose_c(h, m, mp, p)
    return -u_func(c, h, mp)

def solve_housing(m, mp, print_sol=True, p=1):


    # Call optimizer  
    sol = optimize.minimize_scalar(value_of_choice, bounds=None,
                                args=(m, mp, p))

    if print_sol:
        print_solution(sol, m, mp, p)

    # Unpack solution
    h = sol.x
    c = choose_c(h, m, mp, p)
    u = u_func(c, h, mp)
    return c, h, u


def tax_revenues(mp, ms, p=1):


    h_star = np.empty((len(ms),))
    tax_revenue = np.empty((len(ms),))

    for i,m in enumerate(ms):
        c, h, u = solve_housing(m, mp, print_sol=False, p=p)
        h_star[i] = h
        tax_revenue[i] = tau(h, mp)

    return tax_revenue, h_star

def print_solution(sol, m, mp, p=1):
  

    h = sol.x
    c = choose_c(h, m, mp, p)
    u = u_func(c, h, mp)

    # Print
    print(f'c          = {c:6.3f}')
    print(f'h          = {h:6.3f}')
    print(f'user_costs = {user_cost(h, mp, p):6.3f}')
    print(f'u          = {u:6.3f}')
    print(f'm - user_costs - c = {m - user_cost(h, mp, p) - c:.4f}')

