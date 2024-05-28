import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import netket as nk
import jax.numpy as jnp
#import jax
from netket.operator.spin import sigmax,sigmaz,sigmay 
from scipy.sparse.linalg import eigsh
from matplotlib import pyplot as plt
import flax.linen as nn

def H_pl_ising(K, delta, pl_int):
    hamiltonian = nk.operator.LocalOperator(hi)
    for site in g.nodes():
        hamiltonian = hamiltonian - delta * sigmax(hi, site)
        #if site<(N-1):
        i, j, k, l = pl_int[site]
        hamiltonian = hamiltonian - K * sigmaz(hi, i)@sigmaz(hi, j)@sigmaz(hi, k)@sigmaz(hi, l)
    return hamiltonian

def sq_in(Lx,Ly,N):
    pl_in = [[i,(i+1)%N,(i+Lx)%N,(i+Lx+1)%N] for i in range(Lx*Ly)]
    for j in range(1,Ly+1):
        pl_in[j*Lx-1][-1] = pl_in[j*Lx-1][-1] - 2*Lx if(j<Ly-1) else pl_in[j*Lx-1][-1] + (Ly-2)*Lx
    return pl_in

class FFN(nn.Module):
    alpha : int = 1
    @nn.compact
    def __call__(self, x):
        dense = nn.Dense(features=self.alpha * x.shape[-1])
        y = dense(x)
        y = nn.relu(y)
        return jnp.sum(y, axis=-1)

Lx, Ly = 4, 4
N = Lx*Ly
g = nk.graph.Hypercube(length=Lx, n_dim=2, pbc=True)

x = sq_in(Lx,Ly,N)
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

grid = 20
theta = jnp.linspace(0,jnp.pi/2,grid)
e_gs0 = jnp.zeros(grid)
for i in range(grid):
    hamiltonian =  H_pl_ising(jnp.cos(theta[i]), jnp.sin(theta[i]), x)
    model = FFN(alpha=1)
    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=10016)

    # Then we create an optimiser from the standard library.
    # You can also use optax.
    optimizer = nk.optimizer.Sgd(learning_rate=0.05)

    # build the optimisation driver
    gs = nk.driver.VMC(hamiltonian, optimizer, variational_state=vstate,preconditioner=nk.optimizer.SR(diag_shift=0.1))

    # run the driver for 300 iterations. This will display a progress bar
    # by default.

    log=nk.logging.RuntimeLog()
    gs.run(n_iter=100,out=log)
    e_gs0 = e_gs0.at[i].set(log.data["Energy"].Mean[-1])
