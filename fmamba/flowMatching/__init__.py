from .flowMatching import ModelType, PathType, FlowMatching, WeightType
from .flowMatching import Sampler

def create_flow(
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        path_args={},
        t_sample_mode="uniform", # all time steps are euqally likely and are equally weighted and we take monte carlo estimate of the loss over joint distribution of time and pdata, thereby conditional path
):
    """function for creating Flow object
    **Note:** model prediction defaults to velocity
    Args:
    - path_type: type of path to use; default to linear (condOTpath?)
    - learn_score: set model prediction to score
    - learn_noise: set model prediction to noise
    - velocity_weighted: weigh loss by velocity weight
    - likelihood_weighted: weight loss by likelihood weight
    - train_eps: small epsilon for avoiding instability during training
    - sample_eps: small epsilon for aboiding instability during sampling
    """

    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY
    
    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE
    
    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }

    path_type = path_choice[path_type]

    if path_type in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    elif path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else: # velocity & [GVP, LINEAR] is stable everywhere
        # instability usually occurs when using alpha_ratio for score parametrization of vectorfield
        # we directly regress against conditional ut_theta(xt | z)
        # so its all stable and good
        train_eps = 0
        sample_eps = 0
    
    # create flow state
    state = FlowMatching (
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        path_args=path_args,
        t_sample_mode=t_sample_mode,
    )

    return state
    