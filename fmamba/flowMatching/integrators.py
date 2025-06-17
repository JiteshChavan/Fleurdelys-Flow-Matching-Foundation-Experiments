import torch
from torchdiffeq import odeint

class sde:
    """SDE Solver Class"""

    def __init__(
            self,
            drift,
            diffusion,
            *, # all args henceforth are to be passed as keyword args, not positional
            t0,
            t1,
            num_steps,
            sampler_type,
    ):
        assert t0 < t1, "Flow convention is t0 (for pinit) < t1 (for pdata)"

        self.num_timesteps = num_steps
        # TODO: Made a change over here for intuitive interfacing with the class from outside
        # linspace has to be specified num_t_points
        self.t = torch.linspace(t0, t1, num_steps+1)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type
    
    def __Euler_Maruyama_step(self, x, x_mean, t, model, **model_args):
        eps = torch.randn(x.size()).to(x) # w0 for brownian motion
        t = torch.ones(x.size(0)).to(x) * t # expand time step for entire batch
        # variance scales quadratically hence multiply by root dt
        dw = eps * torch.sqrt(self.dt) # dw = N(0, dt*Id), By definition Brownian Motion has Gaussian Increments with linearly scaling variance

        drift = self.drift(x, t, model, **model_args)
        
        # or x_flow (ode trajectory) well not necessarily since score and Vector Field can be expressed as a function of other
        # for gaussian probability paths, so drift, I think accounts for both VectorField and Score Function implicitly
        # hence better referred to signal prior to noise injection as x_mean
        # but since score is direction of high likelihood region, it should in theory work even if we dont inject noise
        x_mean = x + drift * self.dt
        # diffusion is diffusion coefficient, some implicit weird case of root (2 sigma_t)dw instead of variance preserving or exploding SDE sigma_t dw
        # TODO: look into how diffusion coefficient is evaluated
        x = x_mean + torch.sqrt(2* self.diffusion(x, t)) * dw # noise injection
        return x, x_mean
    
    def __Heun_step(self, x, _, t, model, **model_args):
       eps = torch.randn(x.size()).to(x) # eps
       dw = torch.sqrt(self.dt) * eps # N(0, dt Id) gaussian increment
       t_cur = torch.ones(x.size(0)).to(x) * t # expand the time step for all examples in the batch
       diffusion = self.diffusion(x, t_cur) # diffusion coefficient tensor
       xhat = x + torch.sqrt(2*diffusion) * dw   # noise injection beforehand i. restricts the branch of trajectory by isolating randomness away from corrector step while maintaining stochastic nature
       field1 = self.drift(xhat, t_cur, model, **model_args)
       x_intermediate = xhat + self.dt * field1
       field2 = self.drift(x_intermediate, t_cur+self.dt, model, **model_args)
       return xhat + 0.5 * self.dt * (field1 + field2), xhat # at last time point we dont perform the heun step
    
    def __forward_fn(self):
        """TODO: Generalize here by ading all private functions ending with steps to it"""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Sampler type not implemented.")
        
        return sampler
    
    def sample(self, init, model, **model_args):
        """forward loop of sde"""
        x = init
        x_mean = init
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with torch.no_grad():
                x, x_mean = sampler(x, x_mean, ti, model, **model_args)
                samples.append(x)
        return samples

class ode:
    """ODE solver class"""

    def __init__(
            self,
            drift,
            *,
            t0,
            t1,
            sampler_type,
            num_steps,
            atol,
            rtol,
    ):
        assert t0 < t1, "Flow convention is t0 (for pinit) < t1 (for pdata)"

        self.drift = drift
        # TODO: Again same rectification for intuitive interface
        # linspace has to be specified tpoints, not num_steps
        self.t = torch.linspace(t0, t1, num_steps+1)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample (self, x, model, **model_args):
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn (t, x):
            t = torch.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else torch.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_args)
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)
        return samples



        
