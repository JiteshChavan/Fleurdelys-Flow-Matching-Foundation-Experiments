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

        """IMPORTANT: Vector_Field is stable everywhere, when we dont use score parametrization, so the following branches are not 
        that important for our usecase"""

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
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps) # for our particular usecase: vector_field model, ode, and GVP, t0 = 0 & t1= 1
        if self.t_sample_mode == "logitnormal":
            a, b = -0.5, 1
            t = b * torch.randn((x1.shape[0],)) + a
            t = torch.sigmoid(t) * (t1 - t0) + t0
        else:
            t = torch.rand((x1.shape[0],)) * (t1 - t0) + t0 # sample from u[t0,t1) exclusive terminal (for our usecase u[0, 1) )
        t = t.to(x1)
        return t, x0, x1
    
    def training_losses(self, model, x1, model_kwargs=None):
        """
        Loss for training the score model
        Args:
            - model : backbone model; ut_theta or st_theta
            - x1 (z) : z~pdata
            - model_kwargs: additional arguments for the model, y for guidance
        """

        if model_kwargs is None:
            model_kwargs = {}
        
        t, x0, x1 = self.sample(x1) # sample a time point, x0, and data point z
        # Sample from conditional path xt~pt(.|z), evaluate conditional ut_target(xt|z) which is stable for all t~u[0,1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        # get ut_theta(xt|z), to get minimizer theta* which minimizes marginal loss objective as well since expecation over distributions simulated as monte-carlo is lienar
        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape # extract channels and batch size from the conditional prob path.

        assert model_output.size() == (B, *xt.size()[1:-1], C), f"Model Output shape mismatch with the reference point along the conditional prob path"

        terms = {}
        terms["pred"] = model_output
        if self.model_type == ModelType.VELOCITY: # our experiments just train a vector_field (velocity), losses for all time steps are treated equally no time dependent weighting
            terms["loss"] = mean_flat( ((model_output - ut)**2) ) # conditional flow matching loss
        else:
            # returns score_coefficient, x_term in score parametrization of vector_field
            score_coefficient, _ = self.path_sampler.compute_drift(xt,t)
            beta_t, _ = self.path_sampler.compute_beta_t(path.expand_t_like_x(t, xt)) # (B, *dim)
            
            # choice of loss weighting that's dependent or independent of time, for score and denoiser models
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (score_coefficient / beta_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = score_coefficient / (beta_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            """ One way to think about weighting the loss for score and denoiser models is to offset the instability near t = 1
                just one way, not objective reason behind it, one could also experiment with vector_field loss weighting thats time dependent
            """

            if self.model_type == ModelType.NOISE:
                terms["loss"] = mean_flat(weight * ((model_output - x0)**2))
            else: # score model
                # conditional score matching objective is  -eps/beta_t = -x0/beta_t so model_out * beta_t = -x0
                # we multiply by beta_t to avoid division by 0 and thus instability near t=1
                terms["loss"] = mean_flat(weight * ((beta_t * model_output + x0) ** 2))
            
        return terms
    
    def get_drift(self):

    

        
        

