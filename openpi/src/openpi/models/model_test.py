from flax import nnx
import jax
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_load_reinserts_structural_none_bias_leaves_dropped_by_checkpoint_restore():
    import dataclasses

    class _NoBiasModel(_model.BaseModel):
        def __init__(self, rngs: nnx.Rngs):
            super().__init__(action_dim=1, action_horizon=1, max_token_len=1)
            self.proj = nnx.Linear(2, 2, use_bias=False, rngs=rngs)

        def compute_loss(self, rng, observation, actions, *, train=False):  # pragma: no cover
            raise NotImplementedError

        def sample_actions(self, rng, observation):  # pragma: no cover
            raise NotImplementedError

    @dataclasses.dataclass(frozen=True)
    class _NoBiasConfig(_model.BaseModelConfig):
        @property
        def model_type(self) -> _model.ModelType:
            return _model.ModelType.PI0

        def create(self, rng) -> _model.BaseModel:
            return _NoBiasModel(nnx.Rngs(rng))

        def inputs_spec(self, *, batch_size: int = 1):  # pragma: no cover
            raise NotImplementedError

    config = _NoBiasConfig(action_dim=1, action_horizon=1, max_token_len=1)
    params = nnx.state(config.create(jax.random.key(0))).to_pure_dict()
    assert params["proj"]["bias"] is None

    def drop_none_leaves(tree):
        return {
            key: drop_none_leaves(value) if isinstance(value, dict) else value
            for key, value in tree.items()
            if value is not None
        }

    restored = drop_none_leaves(params)
    assert "bias" not in restored["proj"]

    model = config.load(restored)
    assert model.proj.bias.value is None

    missing_kernel = {"proj": {"bias": None}}
    with pytest.raises(ValueError, match="different structure"):
        config.load(missing_kernel, remove_extra_params=False)
