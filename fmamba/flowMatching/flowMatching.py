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
        Loss for training the vector_field model
        Args:
            - model : backbone model; ut_theta or st_theta
            - x1 (z) : z~pdata
            - model_kwargs: additional arguments for the model, y for guidance
        """

        if model_kwargs is None:
            model_kwargs = {}
        
        t, x0, x1 = self.sample(x1) # sample a time point for each example in batch, x0, and data point z
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
            beta_t, _ = self.path_sampler.compute_beta_t(path.expand_t_like_x(t, xt)) # (B, [1]*dim)
            
            # choice of loss weighting that's dependent or independent of time, for score and denoiser models
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (score_coefficient / beta_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = score_coefficient / (beta_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                terms["loss"] = mean_flat(weight * ((model_output - x0)**2))
            else: # score model
                # conditional score matching objective is  -eps/beta_t = -x0/beta_t so model_out * beta_t = -x0
                # we multiply by beta_t to avoid division by 0 and thus instability near t=1
                terms["loss"] = mean_flat(weight * ((beta_t * model_output + x0) ** 2))
            
        return terms
    
    def get_vector_field(self):
        """Returns a function that computes the vector field ut_theta(xt) along the conditional path pt(x|z)""" # gets you vector field function, to which you pass x, t, model, **model_kwargs

        # score model
        def score_ode(x, t, model, **model_kwargs):
            score_coefficient, x_term = self.path_sampler.compute_drift (x, t)
            model_output = model(x, t, **model_kwargs) # score
            vector_field = score_coefficient * model_output + x_term # score parametrization of vector field
            return vector_field
        
        def noise_ode(x, t, model, **model_kwargs):
            score_coefficient, x_term = self.path_sampler.compute_drift(x, t)
            beta_t, _ = self.path_sampler.compute_beta_t(path.expand_t_like_x(t, x)) # beta_t (B, [1]*dims)
            model_output = model(x, t, **model_kwargs) # noise eps
            score = model_output / -beta_t # model output is eps
            return score_coefficient * score + x_term
        
        def velocity_ode (x, t, model, **model_kwargs):
            vector_field = model(x, t, **model_kwargs)
            return vector_field
        
        if self.model_type == ModelType.NOISE:
            vector_field_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            vector_field_fn = score_ode
        else:
            assert self.model_type == ModelType.VELOCITY, f"if model is neither score nor denoiser, it has to be vector_field model"
            vector_field_fn = velocity_ode
        
        def body_fn(x, t, model, **model_kwargs):
            model_output = vector_field_fn (x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, f"Output shape from the vector_field must match input shape"
            return model_output
        
        return body_fn

    
    def get_score(self):
        """ Returns a function that computes the score ∇ₓ log p_t(x|z) for the given model type.
            Handles model types: NOISE, SCORE, VELOCITY.
        """
        
        def score_from_noise_model(x, t, model, **model_kwargs):
            noise = model(x, t, **model_kwargs)
            beta_t, _ = self.path_sampler.compute_beta_t(path.expand_t_like_x(t, x))
            score = - noise / beta_t
            return score

        def score_from_score_model(x, t, model, **model_kwargs):
            score = model (x, t, **model_kwargs)
            return score
        
        def score_from_vector_field (x, t, model, **model_kwargs):
            vector_field = model(x, t, **model_kwargs)
            score = self.path_sampler.get_score_from_vector_field(vector_field, x, t)
            return score


        if self.model_type == ModelType.NOISE:
            score_fn = score_from_noise_model
        elif self.model_type == ModelType.SCORE:
            score_fn = score_from_score_model
        elif self.model_type == ModelType.VELOCITY:
            score_fn = score_from_vector_field
        else:
            raise NotImplementedError()
        
        return score_fn
    
        
        
class Sampler:
    """Sampler class for the flowMatching model"""
    
    def __init__(
            self,
            flowMatching,
        ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - flowMatching: a flowMatching object specify model prdiction type & conditonal path construction (linear, VPCP, GVPCP)
        """

        self.flowMatching = flowMatching
        self.vector_field = flowMatching.get_vector_field()
        self.score = flowMatching.get_score()

    def __get_sde_drift_and_diffusion_coefficient_fn(
            self,
            *,
            diffusion_form="SBDM",
            diffusion_norm=1.0,
    ):
        def diffusion_coefficient_fn (x, t):
            diffusion_coefficient = self.flowMatching.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm) # default diffusion form is None, no diffusion
            return diffusion_coefficient
        
        # drift = vectorfield (x, t, model, model_kwargs) + diffusion_coefficient * score
        sde_drift_fn = lambda x, t, model, **model_kwargs: self.vector_field(x, t, model, **model_kwargs) + diffusion_coefficient_fn(x, t) * self.score(
            x, t, model, **model_kwargs
        )



        sde_diffusion_coefficient_fn = diffusion_coefficient_fn

        return sde_drift_fn, sde_diffusion_coefficient_fn
    
    def __get_last_step(
            self,
            sde_drift,
            *,
            last_step,
            last_step_size,
    ):
        """Get the last step function of the SDE solver"""
        if last_step is None:
            last_step_fn = lambda x, t, model, **model_kwargs: x # identity transformation I * X
        elif last_step == "Mean":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x + sde_drift(x, t, model, **model_kwargs) * last_step_size
            )
        elif last_step == "Tweedie":
            alpha = self.flowMatching.path_sampler.compute_alpha_t # simple aliasing; the original name is too long
            beta = self.flowMatching.path_sampler.compute_beta_t
            # destorys batch structure, either way we arent using this function for last step
            last_step_fn = lambda x, t, model, **model_kwargs: x / alpha(t)[0][0] + (beta(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = lambda x, t, model, **model_kwargs: x + self.vector_field(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()
        
        return last_step_fn
    
    def sample_sde(
            self,
            *,
            sampling_method="Euler",
            diffusion_form="SBDM",
            diffusion_norm=1.0,
            last_step="Mean",
            last_step_size=0.004,
            num_steps=250,
    ):
        """returns a sampling fucntion with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; step along drift ("mean")
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0, 1]
        - num_steps: total integration steps of SDE
        """

        num_steps = num_steps if sampling_method == "Euler" else num_steps // 2
        if last_step is None:
            last_step_size = 0.0
        else:
            if last_step_size == -1:
                last_step_size = 1.0 / num_steps
        
        sde_drift, sde_diffusion_coefficient = self.__get_sde_drift_and_diffusion_coefficient_fn(diffusion_form=diffusion_form, diffusion_norm=diffusion_norm)

        t0, t1 = self.flowMatching.check_interval(
            self.flowMatching.train_eps,
            self.flowMatching.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        # num_steps [0, 1] last step is evaluated at t=1, gnarly, goes against principles
        _sde = sde (sde_drift, sde_diffusion_coefficient, t0=t0, t1=t1, num_steps=num_steps, sampler_type=sampling_method)

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(x0, model, **model_kwargs):
            xs = _sde.sample(x0, model, **model_kwargs)
            t1s = torch.ones(x0.size(0), device=x0.device) * t1
            x = last_step_fn(xs[-1], t1s, model, **model_kwargs)
            xs.append(x)
            
            # num_steps within 0 till 1 and final at t=1
            assert len(xs) == num_steps + 1, "Number of samples along trajectory does not match the number of steps"
            return xs
        
        return _sample
    
    def sample_ode(
            self,
            *,
            sampling_method="dopri5",
            num_steps=50, #50 steps [0,1) + framework to evaluate last step at t=1, last_step size 0 would make it adhere to convention that we evaluate [0,1) and avoid VF eval at t=1
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE;
         default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoint saved during integration; produced by interpolation
        - atol: absolute error tolerance of the solver
        - rtol: relative error tolerance of the solver
        - reverse: inverted time convention or not; default false
        """
        if reverse:
            ut_theta = lambda x, t, model, **model_kwargs: self.vector_field(x, torch.ones_like(t)*(1-t), model, **model_kwargs)

        else:
            ut_theta = self.vector_field
        

        t0, t1 = self.flowMatching.check_interval(
            self.flowMatching.train_eps,
            self.flowMatching.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        # fun fact ode class doesnt care about sampling method
        # despite sending the argument, the odeint solver is never seeded with the method
        # and by default uses dopri15 (5th order Runge-Kutta method)
        _ode = ode (
            drift=ut_theta,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        return _ode.sample
    
    def sample_ode_likelihood(
            self,
            *,
            sampling_method="dopri15",
            num_steps=50,
            atol=1e-6,
            rtol=1e-3,
    ):
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default : Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """

        def _likelihood_drift (x, t, model, **model_kwargs):
            x, _ = x
            eps = torch.randint(2, x.size(), dtype=torch.float, )
            t = torch.ones_like(t) * (1-t)
            with torch.enable_grad():
                x.requires_grad = True
                grad = torch.autograd.grad(torch.sum(self.vector_field(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = torch.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                vector_field = self.vector_field(x, t, model, **model_kwargs)
            return (-vector_field, logp_grad)
            

        t0, t1 = self.flowMatching.check_interval(
            self.flowMatching.train_eps,
            self.flowMatching.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            ut_theta=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = torch.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.flowMatching.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift
        
        return _sample_fn