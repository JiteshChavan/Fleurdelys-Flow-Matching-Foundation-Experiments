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
    
    
