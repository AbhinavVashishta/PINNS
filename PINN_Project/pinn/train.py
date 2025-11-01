# Training loop
import torch
from pinn.loss import loss_fn

def train(pinn_L, pinn_R, data, epochs=5000, lr=1e-3):
    (xL, tL), (xR, tR), (x_ic, t_ic, rho_ic, u_ic), (x_i, t_i), (x_left, x_right, t_b) = data
    
    optimizer = torch.optim.Adam(
        list(pinn_L.parameters()) + list(pinn_R.parameters()), 
        lr=lr
    ) 
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500
    )

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss, (Lp, Li, Lint, Lbc) = loss_fn(
            pinn_L, pinn_R, xL, tL, xR, tR, 
            x_ic, t_ic, rho_ic, u_ic, 
            x_i, t_i, x_left, x_right, t_b
        )
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total={total_loss.item():.4e}, "
                  f"PDE={Lp:.2e}, IC={Li:.2e}, Interface={Lint:.2e}, BC={Lbc:.2e}")

    print("Training done.")