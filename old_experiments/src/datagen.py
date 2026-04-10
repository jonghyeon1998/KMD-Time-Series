import torch

def rk4_step(func, state, dt, **kwargs):
    k1 = func(state, **kwargs)
    k2 = func(state + 0.5 * dt * k1, **kwargs)
    k3 = func(state + 0.5 * dt * k2, **kwargs)
    k4 = func(state + dt * k3, **kwargs)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def lorenz63(xyz, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return torch.stack([dx, dy, dz], dim=-1)

def generate_lorenz63(n_steps=10000, dt=0.01, 
                      sigma=10.0, rho=28.0, beta=8/3,
                      device="cpu"):
    # initial condition
    state = torch.tensor([[1.0, 1.0, 1.0]], device=device)
    traj = [state]
    for _ in range(n_steps):
        state = rk4_step(lorenz63, state, dt, sigma=sigma, rho=rho, beta=beta)
        traj.append(state)
    return torch.cat(traj)

# Example usage
if __name__ == "__main__":
    burnin = 200 
    traj = generate_lorenz63(n_steps=10000+burnin, dt=0.01)
    print(traj.shape)  # (10001, 3)