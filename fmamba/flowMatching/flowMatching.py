import enum

import numpy as np
import torch

from . import path
from .integrators import ode, sde
from .utils import mean_flat

class ModelType(enum.Enum):
    """Model specification"""

    NOISE = enum.auto() # the model predicts epsilon (denoising score matching variant without division by beta_t)
    SCORE = enum.auto() # score model : gradient (log(Pt()))
    VELOCITY = enum.auto() # Vector Field ut_theta(x_t)

class PathType(enum.Enum):
    """
    Path Specification
    """

    LINEAR = enum.auto() # linear flows psi_x0(t) : t[0,1] x ∈ |R d -> |R d
    GVP = enum.auto() # General Variance Preserving Path
    VP = enum.auto() # Variance Preserving Path

class WeightType(enum.Enum):
    """
    Weighting specification
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()

# TODO: mandate consistency after writing the flow matching class wherever it is imported

class FlowMatching:

    def __init__(
            self,
            *,
            model_type,
            path_type,
            loss_type,
            train_eps,
            sample_eps,
            path_args={},
            t_sample_mode = "uniform", # t ~ u [0,1]
            use_blurring=False, # No aggressive blurring near t=0 by default
            blurring_configs={},
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan, # match the enumeration codes with the implemented paths
            PathType.GVP : path.GVPCPlan,
            PathType.VP : path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        # path type construction
        self.path_sampler = path_options[path_type](**path_args)
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.t_sample_mode = t_sample_mode
    
    # TODO: delete if not needed
    # unrelated:
    # the probabilities of a single point in continuous distributions is 0, as the dimensionality increases the normalizing factor increases exponentially
    # log probs are additive, numerically stable and more manageable
    def prior_logp(self, z):
        """
        Standard multivariate normal prior
        Assume z is batched

        Computes log probability of z ~ N(0, Id),
        for each sample in the batch
        """

        shape = torch.tensor(z.size())
        dimensionality = torch.prod(shape[1:]).item() # skip B and product of rest per sample
        def _fn(x): # X: shape (H, W, ...)
            return - dimensionality / 2 *np.log(2* np.pi) - torch.sum(x**2) / 2.0
        
        return torch.vmap(_fn)(z)    # returns shape (B,) log likelihood of all examples under N(0, Id)
    
    def check_interval(
            self,
            train_eps,
            sample_eps,
            *,
            diffusion_form="SBDM", # Score Based Diffusion Model
            sde=False,
            reverse=False,
            eval=False,
            last_step_size=0.0,
    ):
        t0=0
        t1=1 # basically we make no changes if its linear condOT variant construction and ODE and the model is vectorfield lol
        # however we do make changes if its linear CondOT variant construction and its score or noise model (irrespective of ODE SDE)
        eps = train_eps if not eval else sample_eps

        # TODO: mandate consistency this implementation doesnt address t = 0 instability for alpha ratio VPCP
        # TODO: cleaner interface for the path constructions so that each path encapsulates its time end points for stability
        if type(self.path_sampler) in [path.VPCPlan]:
            # if SDE, t1 = 1 - last_step_size, else t1 = 1 - eps
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        elif type(self.path_sampler) in [path.ICPlan, path.GVPCPlan] and (self.model_type != ModelType.VELOCITY or sde):
            # branch for condOT, GVPPath and SDEs or SCORE and NOISE models
            # avoid numerical issue by taking a first semi-implicit step
            # truth is even VPCP is unstable for score parametrization of vector field, this implementation just doesnt address it
            # and specifically score parametrization of VELOCITY of vectorfield is unstable at t= 0
            # I dont understand why this implementation doesnt address velocity model to avoid t0=0

            # this line basically skips t0 = 0 for both SDE, ODE score and noise model if its sde or not for GVPCP and ICP
            # but if its vectorfield/velocity model, it only skips t0 = 0 for "SBDM" and SDE, it wont skip t0 = 0 for velocity ODE
            # even if the alpha ratio is unstable at t = 0
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        
        if reverse:
            t0 = 1 - t0
            t1 = 1 - t1
        return t0, t1
    
    def sample (self, x1):
        """Sampling x0 and t based on the shape of x1 (if needed)
        Args:
            x1 : data point z (B, *dim)
        """

        x0 = torch.randn_like(x1) # eps/x0 ~ pinit
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps) # for our particular usecase: score model and ode, t0 = eps & t1= 1-eps
        if self.t_sample_mode == "logitnormal":
            a, b = -0.5, 1
            t = b * torch.randn((x1.shape[0],)) + a
            t = torch.sigmoid(t) * (t1 - t0) + t0
        else:
            t = torch.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        return t, x0, x1
        

