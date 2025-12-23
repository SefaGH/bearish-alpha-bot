import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.adapters.ppo_trading_adapter import PPOTradingAdapter
from src.ml.ppo.observation_spec import spec_from_feature_columns, build_observation, DEFAULT_EXTRA_FEATURE_NAMES
from src.ml.rl_trading_env import RLTradingEnv
from src.ml.rl_trading_env_gym import RLTradingEnvGym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
import gym
import torch as th


class _FakeMarketDataPipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    async def get_latest_ohlcv(self, symbol: str, timeframe: str):
        return self.df


class _FakeFeaturePipeline:
    def __init__(self, features_df: pd.DataFrame):
        self.features_df = features_df

    def extract_features(self, df: pd.DataFrame, mode: str = "price"):
        return self.features_df


@pytest.mark.asyncio
async def test_observation_parity_fixture():
    fixture_path = Path("tests/fixtures/ppo_parity_fixture.npz")
    assert fixture_path.exists(), "Fixture missing"
    data = np.load(fixture_path, allow_pickle=True)
    features = data["features"]
    prices = data["prices"]
    feature_columns = data["feature_columns"].tolist()
    price_columns = data["price_columns"].tolist()
    timestamps = pd.to_datetime(data["timestamps"])

    features_df = pd.DataFrame(features, columns=feature_columns)
    price_df = pd.DataFrame(prices, columns=price_columns)
    price_df.index = timestamps

    spec = spec_from_feature_columns(feature_columns)

    idx = len(features_df) - 1
    tail_values = {"position_fraction": 0.0, "normalized_pv": 1.0}
    env_obs = build_observation(spec, features_df.iloc[idx], tail_values=tail_values, extra_values={})

    adapter = PPOTradingAdapter(
        {"ppo_enabled": True, "ppo_symbols": ["BTC/USDT:USDT"]},
        market_data_pipeline=_FakeMarketDataPipeline(price_df),
        feature_pipeline=_FakeFeaturePipeline(features_df),
    )
    adapter._spec = spec
    adapter._expected_obs_dim = spec.obs_dim

    state, meta = await adapter._build_state("BTC/USDT:USDT")
    assert state is not None
    assert meta.get("feature_len") == len(feature_columns)
    assert meta.get("extra_len") == 0

    np.testing.assert_allclose(env_obs, state, atol=1e-6)


@pytest.mark.asyncio
async def test_normalized_parity_with_vecnormalize():
    fixture_path = Path("tests/fixtures/ppo_parity_fixture.npz")
    data = np.load(fixture_path, allow_pickle=True)
    features_df = pd.DataFrame(data["features"], columns=data["feature_columns"].tolist())
    price_df = pd.DataFrame(data["prices"], columns=data["price_columns"].tolist())
    price_df.index = pd.to_datetime(data["timestamps"])

    spec = spec_from_feature_columns(features_df.columns)

    # Build env-style obs manually with extras (zeros) and tail defaults
    extra_names = ["extra_ret_1", "extra_ret_3", "extra_range_norm", "extra_vol_10", "extra_trend_ema_ratio"]
    spec_with_extras = spec_from_feature_columns(features_df.columns, extra_feature_names=extra_names)
    feature_row = features_df.iloc[-1]
    tail_values = {"position_fraction": 0.0, "normalized_pv": 1.0}
    extra_values = {name: 0.0 for name in extra_names}
    env_obs = build_observation(spec_with_extras, feature_row, extra_values=extra_values, tail_values=tail_values)

    class _SimpleEnv(gym.Env):
        def __init__(self, obs_dim: int):
            super().__init__()
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(2)
        def reset(self):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        def step(self, action):
            return self.reset(), 0.0, True, False, {}

    vecnorm = VecNormalize(DummyVecEnv([lambda: _SimpleEnv(spec_with_extras.obs_dim)]), norm_obs=True, norm_reward=False, clip_obs=10.0)
    vecnorm.obs_rms.mean = np.zeros((1, spec_with_extras.obs_dim), dtype=np.float32)
    vecnorm.obs_rms.var = np.ones((1, spec_with_extras.obs_dim), dtype=np.float32)
    norm_env_obs = vecnorm.normalize_obs(env_obs[np.newaxis, :])

    adapter = PPOTradingAdapter(
        {"ppo_enabled": True, "ppo_symbols": ["BTC/USDT:USDT"]},
        market_data_pipeline=_FakeMarketDataPipeline(price_df),
        feature_pipeline=_FakeFeaturePipeline(features_df),
    )
    adapter._spec = spec_with_extras
    adapter._expected_obs_dim = spec_with_extras.obs_dim
    adapter._vecnorm = vecnorm

    state, _ = await adapter._build_state("BTC/USDT:USDT")
    assert state is not None
    norm_state = vecnorm.normalize_obs(state[np.newaxis, :])

    np.testing.assert_allclose(norm_env_obs, norm_state, atol=1e-2)


def test_rl_env_with_extras_no_nan():
    fixture_path = Path("tests/fixtures/ppo_parity_fixture.npz")
    data = np.load(fixture_path, allow_pickle=True)
    features_df = pd.DataFrame(data["features"], columns=data["feature_columns"].tolist())
    price_df = pd.DataFrame(data["prices"], columns=data["price_columns"].tolist())
    price_df.index = pd.to_datetime(data["timestamps"])

    spec = spec_from_feature_columns(features_df.columns, extra_feature_names=DEFAULT_EXTRA_FEATURE_NAMES)
    env = RLTradingEnvGym(features_df=features_df, raw_df=price_df, observation_spec=spec)

    obs, _ = env.reset()
    assert not np.isnan(obs).any()
    obs2, _, _, _, _ = env.step(1)
    assert not np.isnan(obs2).any()


@pytest.mark.asyncio
async def test_probabilities_vary_with_vecnormalize():
    model_path = Path("artifacts/ppo/ppo_trading_agent.zip")
    if not model_path.exists():
        pytest.skip("PPO model artifact not found")

    fixture_path = Path("tests/fixtures/ppo_parity_fixture.npz")
    data = np.load(fixture_path, allow_pickle=True)
    features_df = pd.DataFrame(data["features"], columns=data["feature_columns"].tolist())
    price_df = pd.DataFrame(data["prices"], columns=data["price_columns"].tolist())
    price_df.index = pd.to_datetime(data["timestamps"])
    # Build spec matching model obs_dim (include extras if needed)
    extra_names = ["extra_ret_1", "extra_ret_3", "extra_range_norm", "extra_vol_10", "extra_trend_ema_ratio"]
    spec = spec_from_feature_columns(features_df.columns, extra_feature_names=extra_names)

    class _SimpleEnv(gym.Env):
        def __init__(self, obs_dim: int):
            super().__init__()
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(2)
        def reset(self):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        def step(self, action):
            return self.reset(), 0.0, True, False, {}

    vecnorm = VecNormalize(DummyVecEnv([lambda: _SimpleEnv(spec.obs_dim)]), norm_obs=True, norm_reward=False, clip_obs=10.0)
    vecnorm.obs_rms.mean = np.zeros((1, spec.obs_dim), dtype=np.float32)
    vecnorm.obs_rms.var = np.ones((1, spec.obs_dim), dtype=np.float32)

    adapter = PPOTradingAdapter(
        {"ppo_enabled": True, "ppo_symbols": ["BTC/USDT:USDT"]},
        market_data_pipeline=_FakeMarketDataPipeline(price_df),
        feature_pipeline=_FakeFeaturePipeline(features_df),
    )
    adapter._spec = spec
    adapter._expected_obs_dim = spec.obs_dim
    adapter._vecnorm = vecnorm

    model = PPO.load(str(model_path))

    probs = []
    for idx in [50, 100, 150, 200, 250]:
        adapter.market_data_pipeline.df = price_df.iloc[: idx + 1]
        adapter.feature_pipeline.features_df = features_df.iloc[: idx + 1]
        state, _ = await adapter._build_state("BTC/USDT:USDT")
        if state is None:
            continue
        norm_state = vecnorm.normalize_obs(state[np.newaxis, :])
        with np.errstate(all="ignore"), th.no_grad():
            obs_tensor, _ = model.policy.obs_to_tensor(norm_state)
            dist = model.policy.get_distribution(obs_tensor)
            probs_arr = dist.distribution.probs.detach().cpu().numpy()[0]
        probs.append(float(probs_arr[1]))

    assert len(probs) >= 2
    assert np.std(probs) > 1e-6, f"p_long appears constant: {probs}"
