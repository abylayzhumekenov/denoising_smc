"""Burgers-equation model adapter: ground-truth/network loading, sensor mask, and the batched
PDE-residual + observation loss, shared by every proposal x weighting combination that targets
this PDE.

Canonical, single source of truth. Previously this same loss was duplicated (and drifting)
across three places: scripts/generate_burgers.py's get_burger_loss (unbatched, hardcoded
`.view(1,1,128,128)`), scripts/generate_burgers_gem.py's get_burger_loss_batched (a batched
rewrite for N>1 particles), and smc/scripts_2/v_tau.py's burgers_ell_fn (which reached back into
generate_burgers.py to reuse the unbatched one). All call sites now import burger_loss from here
instead. N=1 reduces to exactly the old unbatched behavior -- every existing caller computes
torch.norm(...) over the whole tensor regardless of the extra leading batch dim, so the numeric
result is unaffected by the consolidation.

Adding a new PDE model means writing a sibling module (models/darcy.py, etc.) exposing the same
shape of interface -- load_ground_truth / load_network / a batched (pde_loss, observation_loss)
function -- without touching smc/scripts_2/proposals/ or smc/scripts_2/weightings/ at all.
"""

import pickle

import scipy.io
import torch
import torch.nn.functional as F

from torch_utils.misc import auto_device


def random_sensor(k, grid_size, seed=0, device=None):
    """Return an index mask with k sensor columns randomly placed in a grid_size x grid_size grid."""
    if device is None:
        device = auto_device()
    torch.manual_seed(seed)
    index = torch.zeros(grid_size, grid_size, dtype=torch.float64, device=device)
    known_index = torch.randperm(grid_size, device=device)[:k]
    for i in known_index:
        index[:, i] = 1
    return index


def load_ground_truth(datapath, offset, device=None):
    """Load the offset-th Burgers ground-truth solution field from the .mat test set."""
    if device is None:
        device = auto_device()
    data = scipy.io.loadmat(datapath)
    return torch.tensor(data['output'][offset, :, :], dtype=torch.float64, device=device)


def load_network(network_pkl, device=None):
    """Unpickle the EMA-weights pretrained Burgers denoiser."""
    if device is None:
        device = auto_device()
    with open(network_pkl, 'rb') as f:
        return pickle.load(f)['ema'].to(device)


def burger_loss(u, u_GT, mask, device=None):
    """Batched Burgers PDE-residual and observation loss.

    u, u_GT: [N, 128, 128] or [N, 1, 128, 128] (u_GT may also be unbatched [128, 128] --
    broadcasts against every particle). Returns (pde_loss, observation_loss), each [N, 128, 128],
    observation_loss already masked.
    """
    if device is None:
        device = auto_device()
    N = u.shape[0]
    u = u.reshape(N, 1, 128, 128)
    u_GT = u_GT.reshape(1, 1, 128, 128) if u_GT.dim() == 2 else u_GT.reshape(-1, 1, 128, 128)
    deriv_t = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / 2
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / 2
    u_t = F.conv2d(u, deriv_t, padding=(1, 0))
    u_x = F.conv2d(u, deriv_x, padding=(0, 1))
    u_xx = F.conv2d(u_x, deriv_x, padding=(0, 1))

    pde_loss = (u_t + u * u_x - 0.01 * u_xx).squeeze(1)          # [N, 128, 128]
    observation_loss = (u - u_GT).squeeze(1)                      # [N, 128, 128], broadcasts u_GT
    observation_loss = observation_loss * mask
    return pde_loss, observation_loss


def burgers_ell_fn(net, ground_truth, mask, zeta_obs, device=None):
    """Build the ell_fn(x, sigma_t) -> (ell, s_theta) closure the Doob/V_tau weighting scheme
    (smc/scripts_2/weightings/doob_vtau.py) needs, matching the observation term used in
    scripts/generate_burgers.py's guidance step.
    """
    def ell_fn(x, sigma_t):
        x_N_raw = net(x, sigma_t, class_labels=None).to(torch.float64)
        s_theta = (x - x_N_raw) / sigma_t ** 2
        x_N = x_N_raw * 1.415
        _, observation_loss = burger_loss(x_N, ground_truth, mask, device)
        L_obs = torch.norm(observation_loss, 2) / (128 * 5)
        ell = -zeta_obs * L_obs
        return ell, s_theta

    return ell_fn
