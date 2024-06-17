from pathlib import Path
import random
import time
import traceback
from typing import Any


import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf
import torch
from torch import Tensor, nn, optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import wandb

from corridor_grid.envs import DoorCorridorEnv
from neural_dnf import NeuralDNF, NeuralDNFEO, NeuralDNFMutexTanh
from neural_dnf.neural_dnf import BaseNeuralDNF  # for type hinting
from neural_dnf.utils import DeltaDelayedExponentialDecayScheduler

from common import init_params
from utils import post_to_discord_webhook


PPO_MODEL_DIR = Path(__file__).parent / "dc_ppo_storage/"
if not PPO_MODEL_DIR.exists() or not PPO_MODEL_DIR.is_dir():
    PPO_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class DCPPOBaseAgent(nn.Module):
    """
    To create a base agent, pass in the following parameters:
    - action_size (int): the number of actions the agent can take
    - img_obs_space (gym.spaces.Box): the observation space of the environment
    - image_encoder (nn.Module): the image encoder for the agent
        This should be created using the either of the 2 static methods --
        `create_default_image_encoder()` or `customise_image_encoder()`
    - embedding_size (int): the size of the embedding after the image encoder
        This is calculated and return by the 2 static methods mentioned above.
    - actor (nn.Module | None): an actor network.
        By default, this is None, and will be created using the
        `_create_default_actor()` method.
    - extra_layer (nn.Module | None): an optional extra linear layer (with an
        activation function) to add after the image encoder
        By default, this is None. If you want to add an extra layer, you can
        pass in a nn.Module, or create one using the `customise_image_encoder()`
        method.
    """

    # Model components
    image_encoder: nn.Module
    extra_layer: nn.Module | None
    actor: nn.Module
    critic: nn.Module

    # Model parameters
    action_size: int
    embedding_size: int  # the input size for actor and critic

    # Observation space
    img_obs_space: gym.spaces.Box

    def __init__(
        self,
        action_size: int,
        img_obs_space: gym.spaces.Box,
        image_encoder: nn.Module,
        embedding_size: int,
        actor: nn.Module | None = None,
        extra_layer: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.action_size = action_size
        self.img_obs_space = img_obs_space
        self.image_encoder = image_encoder
        self.extra_layer = extra_layer
        self.embedding_size = embedding_size

        if actor is not None:
            self.actor = actor
        else:
            self.actor = self._create_default_actor()

        self.critic = self._create_default_critic()

        self._init_params()

    def get_value(self, preprocessed_obs: dict[str, Tensor]) -> Tensor:
        """
        Return the value of the state.
        This function is used in PPO algorithm
        """
        embedding = self._get_embedding(preprocessed_obs)
        return self.critic(embedding)

    def get_action_and_value(
        self, preprocessed_obs: dict[str, Tensor], action=None
    ) -> tuple[Tensor, Any, Any, Tensor]:
        """
        Return the action, log probability of the action, entropy of the action
        distribution, and the value of the state.
        This function is used in PPO algorithm
        """
        embedding = self._get_embedding(preprocessed_obs)
        logits = self.actor(embedding)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(embedding),
        )

    def get_actions(
        self, preprocessed_obs: dict[str, Tensor], use_argmax: bool = True
    ):
        """
        Return the actions based on the observation.
        """
        embedding = self._get_embedding(preprocessed_obs)
        logits = self.actor(embedding)
        dist = torch.distributions.Categorical(logits=logits)

        actions = dist.probs.max(dim=1)[1] if use_argmax else dist.sample()  # type: ignore
        return actions.cpu().numpy()

    def _get_embedding(self, preprocessed_obs: dict[str, Tensor]) -> Tensor:
        x = preprocessed_obs["image"].transpose(1, 3).transpose(2, 3)
        embedding = self.image_encoder(x)
        embedding = embedding.reshape(embedding.shape[0], -1)

        if self.extra_layer is not None:
            embedding = self.extra_layer(embedding)

        return embedding

    @staticmethod
    def create_default_image_encoder(
        img_obs_space: gym.spaces.Box,
    ) -> tuple[nn.Module, int]:
        """
        Create the default image encoder for the agent. This method does not
        create an extra linear layer after the convolutional layers.
        Return the image encoder and the final embedding size.
        """
        # Using the sync vector env the box dimension becomes:
        # num_envs x h x w x c
        h = img_obs_space.shape[1]
        w = img_obs_space.shape[2]
        embedding_size = ((h - 1) // 2 - 2) * ((w - 1) // 2 - 2) * 64
        return (
            nn.Sequential(
                nn.Conv2d(3, 16, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
            ),
            embedding_size,
        )

    @staticmethod
    def customise_image_encoder(
        encoder_output_chanel: int = 4,
        kernel_size: int = 2,
        last_act: str = "relu",
        use_extra_layer: bool = False,
        extra_layer_out: int | None = None,
        extra_layer_use_bias: bool = True,
        agent_view_size: int = 3,
    ) -> tuple[nn.Module, nn.Module | None, int]:
        """
        Customise the image encoder for the agent.
        The image encoder consists of a convolutional layer, and an optional
        extra linear layer can be added after the convolutional layer if
        `use_extra_layer` is set to True.
        Return the image encoder, extra layer if using, and the final embedding
        size.
        """
        encoder_modules = []
        encoder_modules.append(
            nn.Conv2d(2, encoder_output_chanel, (kernel_size, kernel_size))
        )
        encoder_modules.append(nn.Tanh() if last_act == "tanh" else nn.ReLU())
        embedding_size = (
            agent_view_size - kernel_size + 1
        ) ** 2 * encoder_output_chanel

        final_embedding_size = extra_layer_out or embedding_size

        if use_extra_layer:
            extra_layer = nn.Sequential(
                nn.Linear(
                    embedding_size,
                    final_embedding_size,
                    bias=extra_layer_use_bias,
                ),
                nn.Tanh(),
            )
        else:
            extra_layer = None

        return (
            nn.Sequential(*encoder_modules),
            extra_layer,
            final_embedding_size,
        )

    def _create_default_actor(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_size),
        )

    def _create_default_critic(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.embedding_size, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def _init_params(self) -> None:
        self.apply(init_params)


class DCPPOMLPAgent(DCPPOBaseAgent):
    """
    An agent for door corridor environment, with MLP actor.
    To create a `DCPPOMLP` agent, pass in the following parameters:
    - action_size (int): the number of actions the agent can take
    - img_obs_space (gym.spaces.Box): the observation space of the environment
    - image_encoder (nn.Module): the image encoder for the agent
        This should be created using the either of the 2 static methods --
        `create_default_image_encoder()` or `customise_image_encoder()`
    - embedding_size (int): the size of the embedding after the image encoder
        This is calculated and return by the 2 static methods mentioned above.
    - actor (nn.Module | None): an actor network.
        By default, this is None, and will be created using the
        `_create_default_actor()` method.
    - extra_layer (nn.Module | None): an optional extra linear layer (with an
        activation function) to add after the image encoder
        By default, this is None. If you want to add an extra layer, you can
        pass in a nn.Module, or create one using the `customise_image_encoder()`
        method.
    """


class DCPPONDNFBasedAgent(DCPPOBaseAgent):
    """
    Base class for agents using a neural DNF module as the actor.
    """

    actor: BaseNeuralDNF

    def __init__(
        self,
        action_size: int,
        img_obs_space: gym.spaces.Box,
        image_encoder: nn.Module,
        embedding_size: int,
        actor: BaseNeuralDNF,
        extra_layer: nn.Module | None = None,
    ) -> None:
        super().__init__(
            action_size,
            img_obs_space,
            image_encoder,
            embedding_size,
            actor,
            extra_layer,
        )

    def get_aux_loss(
        self, preprocessed_obs: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """
        Return the auxiliary loss dictionary for the agent.
        The keys are:
        - l_emb_dis: embedding discretisation loss
        - l_disj_l1_mod: disjunction weight regularisation loss
        """
        embedding = self._get_embedding(preprocessed_obs)

        # Embedding discretisation loss
        l_emb_dis = torch.mean(torch.abs(torch.abs(embedding) - 1))

        # Disjunction weight regularisation loss
        p_t = torch.cat(
            [p.view(-1) for p in self.actor.disjunctions.parameters()]
        )
        l_disj_l1_mod = torch.abs(p_t * (6 - torch.abs(p_t))).mean()

        return {
            "l_emb_dis": l_emb_dis,
            "l_disj_l1_mod": l_disj_l1_mod,
        }

    def get_img_encoding(
        self,
        preprocessed_obs: dict[str, Tensor],
        discretise_img_encoding: bool = False,
    ) -> Tensor:
        """
        Return the image encoding of the observation.
        This function should only be called during evaluation.
        """
        assert (
            not self.training
        ), "get_img_encoding() should only be called during evaluation!"
        with torch.no_grad():
            embedding = self._get_embedding(preprocessed_obs)
            if discretise_img_encoding:
                embedding = torch.sign(embedding)
        return embedding

    def get_actor_output(
        self,
        preprocessed_obs: dict[str, Tensor],
        discretise_img_encoding: bool = False,
    ) -> Tensor:
        """
        Return the raw output of the actor (before tanh)
        This function should only be called during evaluation.
        """
        assert (
            not self.training
        ), "get_actor_output() should only be called during evaluation!"

        with torch.no_grad():
            embedding = self._get_embedding(preprocessed_obs)
            if discretise_img_encoding:
                embedding = torch.sign(embedding)
            return self.actor(embedding)

    def get_actions(
        self,
        preprocessed_obs: dict[str, Tensor],
        use_argmax: bool = True,
        discretise_img_encoding: bool = False,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """
        This function should only be called during evaluation.
        Because of the use of neural DNF module, the output of the actor can be
        treated as a symbolic output after tanh. This function returns both the
        probabilistic/argmax based action and the tanh action.
        """
        assert (
            not self.training
        ), "get_actions() should only be called during evaluation!"

        with torch.no_grad():
            raw_actions = self.get_actor_output(
                preprocessed_obs, discretise_img_encoding
            )
        dist = torch.distributions.Categorical(logits=raw_actions)
        if use_argmax:
            actions = dist.probs.max(1)[1]  # type: ignore
        else:
            actions = dist.sample()
        tanh_action = torch.tanh(raw_actions)

        return actions.cpu().numpy(), tanh_action.cpu().numpy()

    @staticmethod
    def customise_dnf_actor(
        embedding_size: int,
        num_actions: int,
        num_conjunctions: int | None = None,
        use_eo: bool = False,
        use_mt: bool = False,
    ) -> BaseNeuralDNF:
        num_conjunctions = num_conjunctions or num_actions * 3
        assert not (use_eo and use_mt), "Only one of EO and MT can be used!"
        if use_eo:
            actor_class = NeuralDNFEO
        elif use_mt:
            actor_class = NeuralDNFMutexTanh
        else:
            actor_class = NeuralDNF
        return actor_class(embedding_size, num_conjunctions, num_actions, 1.0)


class DCPPONDNFAgent(DCPPONDNFBasedAgent):
    """
    An agent for door corridor environment, with `NeuralDNF` as actor.
    This agent is not expected to use as a training agent, but used as a
    post-training evaluation agent from either a trained `DCPPONDNFEOAgent` or
    `DCPPONDNFMutexTanhAgent`.
    """

    actor: NeuralDNF

    def __init__(
        self,
        action_size: int,
        img_obs_space: gym.spaces.Box,
        image_encoder: nn.Module,
        embedding_size: int,
        actor: NeuralDNF,
        extra_layer: nn.Module | None = None,
    ) -> None:
        super().__init__(
            action_size,
            img_obs_space,
            image_encoder,
            embedding_size,
            actor,
            extra_layer,
        )


class DCPPONDNFEOAgent(DCPPONDNFBasedAgent):
    """
    An agent for door corridor environment, with `NeuralDNFEO` actor.
    This agent is used for training, and to be converted to a `DCPPONDNFAgent`
    for post-training evaluation.
    To create a `NeuralDNFEO` agent, pass in the following parameters:
    - action_size (int): the number of actions the agent can take
    - img_obs_space (gym.spaces.Box): the observation space of the environment
    - image_encoder (nn.Module): the image encoder for the agent
        This should be created using the either of the 2 static methods --
        `create_default_image_encoder()` or `customise_image_encoder()`
    - embedding_size (int): the size of the embedding after the image encoder
        This is calculated and return by the 2 static methods mentioned above.
    - actor (NeuralDNFEO): an `NeuralDNFEO` actor.
        This should be created using the `customise_dnf_actor()` method.
    - extra_layer (nn.Module | None): an optional extra linear layer (with an
        activation function) to add after the image encoder
        By default, this is None. If you want to add an extra layer, you can
        pass in a nn.Module, or create one using the `customise_image_encoder()`
        method.
    """

    actor: NeuralDNFEO

    def __init__(
        self,
        action_size: int,
        img_obs_space: gym.spaces.Box,
        image_encoder: nn.Module,
        embedding_size: int,
        actor: NeuralDNFEO,
        extra_layer: nn.Module | None = None,
    ) -> None:
        super().__init__(
            action_size,
            img_obs_space,
            image_encoder,
            embedding_size,
            actor,
            extra_layer,
        )

    def to_ndnf_agent(self) -> DCPPONDNFAgent:
        """
        Convert this agent to a DCPPONDNFAgent.
        """
        return DCPPONDNFAgent(
            self.action_size,
            self.img_obs_space,
            self.image_encoder,
            self.embedding_size,
            self.actor.to_ndnf(),
            self.extra_layer,
        )

    def get_aux_loss(
        self, preprocessed_obs: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """
        Return the auxiliary loss dictionary for the agent.
        The keys are:
        - l_emb_dis: embedding discretisation loss
        - l_disj_l1_mod: disjunction weight regularisation loss
        - l_tanh_conj: tanh conjunction output regularisation loss
        """
        aux_loss_dict = super().get_aux_loss(preprocessed_obs)

        embedding = self._get_embedding(preprocessed_obs)

        # Push tanhed conjunction output towards -1 and 1 only
        tanh_conj = torch.tanh(self.actor.conjunctions(embedding))
        l_tanh_conj = (1 - tanh_conj.abs()).mean()

        return {**aux_loss_dict, "l_tanh_conj": l_tanh_conj}


class DCPPONDNFMutexTanhAgent(DCPPONDNFBasedAgent):
    """
    An agent for door corridor environment, with `NeuralDNFMutexTanh` actor.
    This agent is used for training. It can be converted to a `DCPPONDNFAgent`
    for post-training evaluation, or used directly for evaluation.
    To create a `DCPPONDNFMutexTanhAgent` agent, pass in the following parameters:
    - action_size (int): the number of actions the agent can take
    - img_obs_space (gym.spaces.Box): the observation space of the environment
    - image_encoder (nn.Module): the image encoder for the agent
        This should be created using the either of the 2 static methods --
        `create_default_image_encoder()` or `customise_image_encoder()`
    - embedding_size (int): the size of the embedding after the image encoder
        This is calculated and return by the 2 static methods mentioned above.
    - actor (NeuralDNFMutexTanh): a `NeuralDNFMutexTanh` actor
        This should be created using the `customise_dnf_actor()` method.
    - extra_layer (nn.Module | None): an optional extra linear layer (with an
        activation function) to add after the image encoder
        By default, this is None. If you want to add an extra layer, you can
        pass in a nn.Module, or create one using the `customise_image_encoder()`
        method.
    """

    actor: NeuralDNFMutexTanh

    def __init__(
        self,
        action_size: int,
        img_obs_space: gym.spaces.Box,
        image_encoder: nn.Module,
        embedding_size: int,
        actor: NeuralDNFMutexTanh,
        extra_layer: nn.Module | None = None,
    ) -> None:
        super().__init__(
            action_size,
            img_obs_space,
            image_encoder,
            embedding_size,
            actor,
            extra_layer,
        )

    def get_action_and_value(
        self, preprocessed_obs: dict[str, Tensor], action=None
    ) -> tuple[Tensor, Any, Any, Tensor]:
        """
        Return the action, log probability of the action, entropy of the action
        distribution, and the value of the state.
        This function is used in PPO algorithm
        """
        embedding = self._get_embedding(preprocessed_obs)
        logits = self.actor(embedding)
        dist = torch.distributions.Categorical(probs=(logits + 1) / 2)

        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(embedding),
        )

    def get_actor_output(
        self,
        preprocessed_obs: dict[str, Tensor],
        discretise_img_encoding: bool = False,
        raw_output: bool = True,
        mutex_tanh: bool = False,
    ) -> Tensor:
        """
        Return the raw output of the `NeuralDNFMutexTanh` actor:
        - `raw_output` True: return the raw logits
        - `mutex_tanh` True: return the mutex-tanhed output
        This function should only be called during evaluation.
        """
        assert raw_output or mutex_tanh, "At least one of raw_output and "
        "mutex_tanh should be True!"

        assert not (raw_output and mutex_tanh), "Only one of raw_output and "
        "mutex_tanh can be True!"

        with torch.no_grad():
            embedding = self._get_embedding(preprocessed_obs)
            if discretise_img_encoding:
                embedding = torch.sign(embedding)
        if raw_output:
            return self.actor.get_raw_output(embedding)
        return self.actor(embedding)

    def get_actions(
        self,
        preprocessed_obs: dict[str, Tensor],
        use_argmax: bool = True,
        discretise_img_encoding: bool = False,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """
        This function should only be called during evaluation.
        Because of the use of neural DNF module, the output of the actor can be
        treated as a symbolic output after tanh. This function returns both the
        probabilistic/argmax based action and the tanh action.
        """
        assert (
            not self.training
        ), "get_actions() should only be called during evaluation!"

        with torch.no_grad():
            embedding = self._get_embedding(preprocessed_obs)
            if discretise_img_encoding:
                embedding = torch.sign(embedding)
            act = self.actor(embedding)
            dist = torch.distributions.Categorical(probs=(act + 1) / 2)
            tanh_actions = torch.tanh(self.actor.get_raw_output(embedding))

        actions = dist.probs.max(dim=1)[1] if use_argmax else dist.sample()  # type: ignore
        tanh_actions = torch.tanh(self.actor.get_raw_output(embedding))
        return (
            actions.detach().cpu().numpy(),
            tanh_actions.detach().cpu().numpy(),
        )

    def get_aux_loss(
        self, preprocessed_obs: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """
        Return the auxiliary loss dictionary for the agent.
        The keys are:
        - l_emb_dis: embedding discretisation loss
        - l_disj_l1_mod: disjunction weight regularisation loss
        - l_tanh_conj: tanh conjunction output regularisation loss
        - l_mt_ce2: mutux tanh auxiliary loss
        """
        aux_loss_dict = super().get_aux_loss(preprocessed_obs)

        embedding = self._get_embedding(preprocessed_obs)

        # Push tanhed conjunction output towards -1 and 1 only
        tanh_conj = torch.tanh(self.actor.conjunctions(embedding))
        l_tanh_conj = (1 - tanh_conj.abs()).mean()

        act_out = self.actor(embedding)
        tanh_out = torch.tanh(self.actor.get_raw_output(embedding))

        p_k = (act_out + 1) / 2
        p_k_hat = (tanh_out + 1) / 2
        l_mt_ce2 = -torch.sum(
            p_k * torch.log(p_k_hat + 1e-8)
            + (1 - p_k) * torch.log(1 - p_k_hat + 1e-8)
        )

        return {
            **aux_loss_dict,
            "l_tanh_conj": l_tanh_conj,
            "l_mt_ce2": l_mt_ce2,
        }


def construct_model(
    cfg: DictConfig,
    num_actions: int,
    use_dnf: bool,
    img_obs_space: gym.spaces.Box,
) -> DCPPOBaseAgent:
    if "customised_image_encoder" in cfg:
        (
            img_encoder,
            extra_layer,
            embedding_size,
        ) = DCPPOBaseAgent.customise_image_encoder(
            **cfg["customised_image_encoder"]
        )
    else:
        (
            img_encoder,
            embedding_size,
        ) = DCPPOBaseAgent.create_default_image_encoder(img_obs_space)
        extra_layer = None

    if "customised_actor" in cfg:
        actor = nn.Sequential(
            nn.Linear(embedding_size, cfg["customised_actor"]["hidden_size"]),
            nn.Tanh(),
            nn.Linear(cfg["customised_actor"]["hidden_size"], num_actions),
        )
    else:
        actor = None

    if use_dnf:
        dnf_actor = DCPPONDNFBasedAgent.customise_dnf_actor(
            embedding_size=embedding_size,
            num_actions=num_actions,
            num_conjunctions=(
                cfg["num_conjunctions"] if "num_conjunctions" in cfg else None
            ),
            use_eo=cfg["use_eo"],
            use_mt=cfg["use_mt"],
        )
        model_class = (
            DCPPONDNFEOAgent
            if cfg["use_eo"]
            else (DCPPONDNFMutexTanhAgent if cfg["use_mt"] else DCPPONDNFAgent)
        )
        return model_class(
            action_size=num_actions,
            img_obs_space=img_obs_space,
            actor=dnf_actor,  # type: ignore
            image_encoder=img_encoder,
            extra_layer=extra_layer,
            embedding_size=embedding_size,
        )

    return DCPPOMLPAgent(
        action_size=num_actions,
        img_obs_space=img_obs_space,
        image_encoder=img_encoder,
        embedding_size=embedding_size,
        actor=actor,
        extra_layer=extra_layer,
    )


def make_env(seed: int, idx: int, capture_video: bool):
    def thunk():
        if capture_video and idx == 0:
            env = DoorCorridorEnv(render_mode="rgb_array")
            video_dir = Path("videos")
            env = RecordVideo(env, str(video_dir.absolute()))
        else:
            env = DoorCorridorEnv()
        env = RecordEpisodeStatistics(env)

        env.action_space.seed(seed)
        return env

    return thunk


def train_ppo(
    training_cfg: DictConfig,
    full_experiment_name: str,
    writer: SummaryWriter,
    save_model: bool = True,
) -> tuple[Path | None, DCPPOBaseAgent]:
    use_dnf = "ndnf" in full_experiment_name
    batch_size = int(training_cfg["num_envs"] * training_cfg["num_steps"])
    minibatch_size = int(batch_size // training_cfg["num_minibatches"])
    num_iterations = int(training_cfg["total_timesteps"] // batch_size)

    use_cuda = torch.cuda.is_available() and training_cfg["use_cuda"]

    use_mps = (
        torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
        and training_cfg.get("use_mps", False)
    )

    if use_mps:
        device = torch.device("mps")
    else:
        device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

    print(f"Training on device: {device}")

    # Env setup
    envs = gym.vector.SyncVectorEnv([make_env(i, i, False) for i in range(8)])
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Set up the model
    agent = construct_model(
        training_cfg,
        DoorCorridorEnv.get_num_actions(),
        use_dnf,
        envs.observation_space["image"],  # type: ignore
    )
    agent.train()
    agent.to(device)

    optimizer = optim.Adam(
        agent.parameters(), lr=training_cfg["learning_rate"], eps=1e-5
    )

    # ALGO Logic: Storage setup
    num_steps: int = training_cfg["num_steps"]
    num_envs: int = training_cfg["num_envs"]

    obs_shape = (num_steps, num_envs) + envs.single_observation_space[  # type: ignore
        "image"
    ].shape
    # obs_shape: (num_steps, num_envs, agent_view_size, agent_view_size, 2)
    obs = torch.zeros(obs_shape).to(device)

    action_shape = (num_steps, num_envs) + envs.single_action_space.shape  # type: ignore
    # action_shape: (num_steps, num_envs)
    actions = torch.zeros(action_shape).to(device)

    log_probs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs_dict, _ = envs.reset()
    next_obs = torch.Tensor(next_obs_dict["image"]).to(device)
    next_obs_dict = {"image": next_obs}
    next_done = torch.zeros(num_envs).to(device)

    if isinstance(agent, DCPPONDNFBasedAgent):
        dds_cfg = training_cfg["dds"]
        dds = DeltaDelayedExponentialDecayScheduler(
            initial_delta=dds_cfg["initial_delta"],
            delta_decay_delay=dds_cfg["delta_decay_delay"],
            delta_decay_steps=dds_cfg["delta_decay_steps"],
            delta_decay_rate=dds_cfg["delta_decay_rate"],
            target_module_type=agent.actor.__class__.__name__,
        )
        agent.actor.set_delta_val(dds_cfg["initial_delta"])

    for iteration in range(1, num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if training_cfg["anneal_lr"]:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lr_now = frac * training_cfg["learning_rate"]
            optimizer.param_groups[0]["lr"] = lr_now

        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(
                    next_obs_dict
                )
                values[step] = value.flatten()
            actions[step] = action
            log_probs[step] = log_prob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_dict, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).float().to(device).view(-1)
            next_obs = torch.Tensor(next_obs_dict["image"]).to(device)
            next_obs_dict = {"image": next_obs}
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        if (
                            iteration % training_cfg["log_interval"] == 0
                            and step == num_steps - 1
                        ):
                            print(
                                f"global_step={global_step}, "
                                f"episodic_return={info['episode']['r']}"
                            )
                        writer.add_scalar(
                            "charts/episodic_return",
                            info["episode"]["r"],
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/episodic_length",
                            info["episode"]["l"],
                            global_step,
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs_dict).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_vals = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_vals = values[t + 1]
                delta = (
                    rewards[t]
                    + training_cfg["gamma"] * next_vals * next_non_terminal
                    - values[t]
                )
                advantages[t] = last_gae_lam = (
                    delta
                    + training_cfg["gamma"]
                    * training_cfg["gae_lambda"]
                    * next_non_terminal
                    * last_gae_lam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape(
            (-1,) + envs.single_observation_space["image"].shape  # type: ignore
        )
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)  # type: ignore
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clip_fracs = []
        for _ in range(training_cfg["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, new_log_prob, entropy, new_val = agent.get_action_and_value(
                    {"image": b_obs[mb_inds]}, b_actions.long()[mb_inds]
                )
                log_ratio = new_log_prob - b_log_probs[mb_inds]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > training_cfg["clip_coef"])
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if training_cfg["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1 - training_cfg["clip_coef"],
                    1 + training_cfg["clip_coef"],
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_val = new_val.view(-1)
                if training_cfg["clip_vloss"]:
                    v_loss_unclipped = (new_val - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_val - b_values[mb_inds],
                        -training_cfg["clip_coef"],
                        training_cfg["clip_coef"],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_val - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - training_cfg["ent_coef"] * entropy_loss
                    + v_loss * training_cfg["vf_coef"]
                )

                optimizer.zero_grad(set_to_none=True)

                if isinstance(agent, DCPPONDNFBasedAgent):
                    aux_loss_dict = agent.get_aux_loss(next_obs_dict)
                    l_emb_dis_lambda = training_cfg["aux_loss"][
                        "emb_dis_lambda"
                    ]
                    l_emb_dis = aux_loss_dict["l_emb_dis"]
                    loss += l_emb_dis_lambda * l_emb_dis

                    l_disj_l1_mod_lambda = training_cfg["aux_loss"][
                        "dis_l1_mod_lambda"
                    ]
                    l_disj_l1_mod = aux_loss_dict["l_disj_l1_mod"]
                    loss += l_disj_l1_mod_lambda * l_disj_l1_mod

                    if isinstance(agent, DCPPONDNFEOAgent):
                        l_tanh_conj_lambda = training_cfg["aux_loss"][
                            "tanh_conj_lambda"
                        ]
                        l_tanh_conj = aux_loss_dict["l_tanh_conj"]
                        loss += l_tanh_conj_lambda * l_tanh_conj

                    if isinstance(agent, DCPPONDNFMutexTanhAgent):
                        l_tanh_conj_lambda = training_cfg["aux_loss"][
                            "tanh_conj_lambda"
                        ]
                        l_tanh_conj = aux_loss_dict["l_tanh_conj"]
                        loss += l_tanh_conj_lambda * l_tanh_conj

                        l_mt_ce2_lambda = training_cfg["aux_loss"][
                            "mt_ce2_lambda"
                        ]
                        l_mt_ce2 = aux_loss_dict["l_mt_ce2"]
                        loss += l_mt_ce2_lambda * l_mt_ce2

                loss.backward()
                nn.utils.clip_grad_norm_(  # type: ignore
                    agent.parameters(), training_cfg["max_grad_norm"]
                )
                optimizer.step()

            if (
                training_cfg["target_kl"] is not None
                and approx_kl > training_cfg["target_kl"]  # type: ignore
            ):
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y  # type: ignore
        )

        if isinstance(agent, DCPPONDNFBasedAgent):
            delta_dict = dds.step(agent.actor)
            new_delta = delta_dict["new_delta_vals"][0]
            old_delta = delta_dict["old_delta_vals"][0]

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)  # type: ignore
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)  # type: ignore
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)  # type: ignore
        writer.add_scalar(
            "losses/old_approx_kl", old_approx_kl.item(), global_step  # type: ignore
        )
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)  # type: ignore
        writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), global_step)
        writer.add_scalar(
            "losses/explained_variance", explained_var, global_step
        )
        if isinstance(agent, DCPPONDNFBasedAgent):
            writer.add_scalar("losses/l_emb_dis", l_emb_dis.item(), global_step)
            writer.add_scalar(
                "losses/l_disj_l1_mod", l_disj_l1_mod.item(), global_step
            )
            writer.add_scalar(
                "losses/l_tanh_conj", l_tanh_conj.item(), global_step
            )

            if isinstance(agent, DCPPONDNFMutexTanhAgent):
                writer.add_scalar(
                    "losses/l_mt_ce2", l_mt_ce2.item(), global_step
                )

        if isinstance(agent, DCPPONDNFBasedAgent):
            writer.add_scalar("charts/delta", old_delta, global_step)  # type: ignore
            if new_delta != old_delta:  # type: ignore
                print(f"i={iteration} old delta={old_delta:.3f}\tnew delta={new_delta:.3f}")  # type: ignore

        writer.add_scalar(
            "charts/SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )

        # If training DCPPONeuralDNFMutexTanhAgent, plot y_k, hat_y_k
        if isinstance(agent, DCPPONDNFMutexTanhAgent):
            ret = plot_y_k_hat_y_k(agent)
            for key, value in ret.items():
                writer.add_scalar(f"mt/{key}", value, global_step)

    envs.close()
    writer.close()

    if save_model:
        model_dir = PPO_MODEL_DIR / full_experiment_name
        if not model_dir.exists() or not model_dir.is_dir():
            model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pth"
        torch.save(agent.state_dict(), model_path)
        return model_path, agent

    return None, agent


def plot_y_k_hat_y_k(model: DCPPONDNFMutexTanhAgent) -> dict[str, Any]:
    obs = {
        "image": torch.Tensor(
            [
                [
                    [[0, 0], [0, 0], [0, 0]],
                    [[2, 0], [2, 0], [2, 0]],
                    [[2, 0], [4, 0], [3, 1]],
                ],  # this is the observation at the first time step
                [
                    [[0, 0], [0, 0], [0, 0]],
                    [[2, 0], [3, 1], [2, 0]],
                    [[2, 0], [4, 0], [2, 0]],
                ],  # this is the second observation if we turn right after the first
            ],
            device=next(model.parameters()).device,
        )
    }

    with torch.no_grad():
        hat_y = torch.tanh(
            model.get_actor_output(
                preprocessed_obs=obs, raw_output=True, mutex_tanh=False
            )
        )
        y = model.get_actor_output(
            preprocessed_obs=obs, raw_output=False, mutex_tanh=True
        )

    ret = {}
    for i in range(len(obs)):
        for j in range(hat_y.shape[1]):
            ret[f"y_{i}_{j}"] = y[i, j].item()
            ret[f"hat_y_{i}_{j}"] = hat_y[i, j].item()

    return ret


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    training_cfg = cfg["training"]
    seed = training_cfg["seed"]
    full_experiment_name = training_cfg["experiment_name"] + f"_{seed}"

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_dir_name = "-".join(
        [
            (s.upper() if i in [0, 1] else s)
            for i, s in enumerate(full_experiment_name.split("_"))
        ]
    )

    use_wandb = cfg["wandb"]["use_wandb"]
    if use_wandb:
        run = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            config=OmegaConf.to_container(
                training_cfg, resolve=True, throw_on_missing=True
            ),  # type: ignore
            dir=HydraConfig.get().run.dir,
            name=run_dir_name,
            tags=cfg["wandb"]["tags"] if "tags" in cfg["wandb"] else [],
            monitor_gym=True,
            save_code=True,
            sync_tensorboard=True,
        )

    # torch.autograd.set_detect_anomaly(True)  # type: ignore

    writer_dir = Path(HydraConfig.get().run.dir) / "tb"
    writer = SummaryWriter(writer_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % (
            "\n".join(
                [
                    f"|{key}|{value}|"
                    for key, value in vars(training_cfg).items()
                ]
            )
        ),
    )

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        model_path, _ = train_ppo(training_cfg, full_experiment_name, writer)
        assert model_path is not None
        if use_wandb:
            wandb.save(glob_str=str(model_path.absolute()))

        if use_discord_webhook:
            msg_body = "Success!"
    except BaseException as e:
        if use_discord_webhook:
            if isinstance(e, KeyboardInterrupt):
                keyboard_interrupt = True
            else:
                msg_body = "Check the logs for more details."

        print(traceback.format_exc())
        errored = True
    finally:
        if use_discord_webhook:
            if msg_body is None:
                msg_body = ""
            webhook_url = cfg["webhook"]["discord_webhook_url"]
            post_to_discord_webhook(
                webhook_url=webhook_url,
                experiment_name=full_experiment_name,
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )
        if use_wandb:
            wandb.finish()
        if not errored:
            path = Path(HydraConfig.get().run.dir)
            path.rename(path.absolute().parent / run_dir_name)


if __name__ == "__main__":
    import multiprocessing as mp

    if mp.get_start_method() != "forkserver":
        mp.set_start_method("forkserver", force=True)

    run_experiment()
