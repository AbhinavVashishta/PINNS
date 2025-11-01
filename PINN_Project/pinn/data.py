# Sampling training data with correct initial conditions
import torch

def generate_collocation_points(N_f=2000):
    #Generate collocation points for PDE residual
    x = torch.linspace(-1, 1, N_f).view(-1, 1)
    t = torch.rand_like(x)
    left_mask = x <= 0
    right_mask = x > 0
    return (x[left_mask], t[left_mask]), (x[right_mask], t[right_mask])

def initial_condition(N_ic=500):
    """
    Initial conditions:
    rho = 0.1 for x ≤ 0, rho = 10 for x > 0
    u = 2 for x ≤ 0, u = 1 for x > 0
    """
    x_ic = torch.linspace(-1, 1, N_ic).view(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    
    # Initial conditions for rho
    rho_ic = torch.where(x_ic <= 0, 
                         torch.full_like(x_ic, 0.1), 
                         torch.full_like(x_ic, 10.0))
    
    # Initial conditions for u
    u_ic = torch.where(x_ic <= 0, 
                       torch.full_like(x_ic, 2.0), 
                       torch.full_like(x_ic, 1.0))
    
    return x_ic, t_ic, rho_ic, u_ic

def interface_points(N_i=200):
    # Points to the interface x=0
    x_i = torch.zeros((N_i, 1))
    t_i = torch.linspace(0, 1, N_i).view(-1, 1)
    return x_i, t_i

def boundary_points(N_b=100):
    # Points at boundaries x=-1 and x=1 for zero-gradient BC
    t_b = torch.linspace(0, 1, N_b).view(-1, 1)
    x_left = torch.full_like(t_b, -1.0)
    x_right = torch.full_like(t_b, 1.0)
    return x_left, x_right, t_b