`scripts/inspect_replay_and_td.py` dosyasını, yeni **State Augmentation** (özellik + pozisyon bilgisi) yapısına uygun olarak güncelledim.

Bu kod, belirtilen veri setini yükler, geçici bir `Environment` oluşturarak gerçek state boyutunu (örn. 84) öğrenir ve ajanı bu doğru boyutla başlatır. Böylece boyut uyuşmazlığı hatası almazsınız.

Aşağıdaki kodu kopyalayıp dosyanın tamamının yerine yapıştırabilirsiniz.

```python
#!/usr/bin/env python3
"""Inspect replay buffer statistics and q-value distributions for RL agents.

Important: This tool expects replay files created by this project. Never load
untrusted pickle inputs - they can execute arbitrary code on deserialization.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Proje kök dizinini yola ekle
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Gerekli Importlar
from src.ml.rl_trading_env import RLTradingEnv
from scripts.rl_dataset_utils import load_npz_dataset
from src.ml.reinforcement_learning import TradingRLAgent


# -----------------------------------------------------------------------------
# Dataset & model helpers
# -----------------------------------------------------------------------------

def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def apply_reward_transform(
    rewards: np.ndarray,
    *,
    clip_value: Optional[float] = None,
    clip_range: Optional[Tuple[float, float]] = None,
    scale: float = 1.0,
) -> np.ndarray:
    if clip_value is not None:
        rewards = np.clip(rewards, -clip_value, clip_value)
    elif clip_range is not None:
        min_limit, max_limit = clip_range
        rewards = np.clip(rewards, min_limit, max_limit)
    if scale != 1.0:
        rewards = rewards * scale
    return rewards


def _extract_checkpoint_state_size(model_path: Path) -> int | None:
    """Checkpoint'ten state boyutunu tahmin etmeye çalışır."""
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception:
        return None

    state_dict = checkpoint.get("q_network")
    if not isinstance(state_dict, dict):
        return None
    
    # İlk katmanın ağırlık matrisini bul (input_features x hidden_size)
    for key, tensor in state_dict.items():
        if key.endswith("network.0.weight") and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            return int(tensor.shape[1]) # shape[1] input size'dır
    return None


def load_agent(model_path: Path, state_size: int) -> Any:
    # Checkpoint'teki boyutu kontrol et
    ckpt_size = _extract_checkpoint_state_size(model_path)
    
    final_size = state_size
    if ckpt_size is not None and ckpt_size != state_size:
        logging.warning(f"⚠️ Dimension Mismatch! Env says {state_size}, Checkpoint says {ckpt_size}.")
        logging.warning(f"➡️ Using Checkpoint size ({ckpt_size}) to prevent crash.")
        final_size = ckpt_size

    logging.info(f"🤖 Loading Agent with state_size={final_size}")
    agent = TradingRLAgent(state_size=final_size, action_size=3)
    
    try:
        agent.load_model(str(model_path))
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise
        
    return agent


def build_replay_from_dataset(dataset_features: Any, close_prices: np.ndarray) -> List[Tuple]:
    """
    Sentetik bir replay buffer oluşturur.
    DİKKAT: Bu fonksiyon sadece feature'ları state olarak kullanır. 
    Eğer model 'Augmented State' (Feature + Pozisyon) bekliyorsa bu replay uyumsuz olabilir.
    Bu durumda state vektörünü sıfırlarla (padding) tamamlamamız gerekebilir.
    """
    replay: List[Tuple] = []
    max_idx = len(dataset_features) - 1
    
    # Not: Sentetik replay oluştururken modelin beklediği ekstra state bilgilerini (pozisyon vb.)
    # tam olarak simüle edemeyebiliriz. Bu fonksiyon genelde basit debug içindir.
    
    for i in range(max_idx):
        state = dataset_features.iloc[i].to_numpy(dtype=float)
        next_state = dataset_features.iloc[i + 1].to_numpy(dtype=float)
        reward = float(close_prices[i + 1] - close_prices[i])
        action = random.randint(0, 2)
        done = i == max_idx - 1
        replay.append((state, action, reward, next_state, done))
    return replay


def load_replay(replay_path: Path) -> List[Tuple]:
    with open(replay_path, "rb") as fh:
        return pickle.load(fh)


def compute_reward_stats(replay: List[Tuple]) -> Dict[str, float]:
    rewards = np.array([exp[2] for exp in replay])
    if rewards.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "1%": 0.0, "99%": 0.0}
    return {
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "min": float(rewards.min()),
        "max": float(rewards.max()),
        "1%": float(np.percentile(rewards, 1)),
        "99%": float(np.percentile(rewards, 99)),
    }


def compute_td_stats(
    agent: Any,
    replay: List[Tuple],
    batch_size: int = 32,
    samples: int = 10,
    reward_clip: Optional[float] = None,
    reward_clip_range: Optional[Tuple[float, float]] = None,
    reward_scale: float = 1.0,
) -> Dict[str, Any]:
    td_errors: List[float] = []
    q_stds: List[float] = []

    if not replay:
        raise ValueError("Replay buffer is empty; cannot compute TD statistics")

    gamma = getattr(agent, "gamma", 0.99)
    scale_fn = getattr(agent, "_scale_q", None)

    # Modelin beklediği input boyutu
    expected_dim = agent.state_size

    for _ in range(samples):
        batch = random.sample(replay, min(batch_size, len(replay)))
        
        # State boyutlarını kontrol et ve gerekirse padle
        raw_states = [exp[0] for exp in batch]
        raw_next_states = [exp[3] for exp in batch]
        
        # Padding mantığı: Eğer replay'deki veri (örn 82) modelden (örn 84) küçükse
        # eksik kısımları 0 ile doldur. (Genelde sentetik replay durumunda olur)
        if raw_states[0].shape[0] < expected_dim:
            diff = expected_dim - raw_states[0].shape[0]
            raw_states = [np.pad(s, (0, diff), 'constant') for s in raw_states]
            raw_next_states = [np.pad(s, (0, diff), 'constant') for s in raw_next_states]
        
        states = torch.FloatTensor(np.stack(raw_states))
        next_states = torch.FloatTensor(np.stack(raw_next_states))
        
        actions = torch.LongTensor([exp[1] for exp in batch])
        reward_array = np.stack([exp[2] for exp in batch])
        reward_array = apply_reward_transform(
            reward_array,
            clip_value=reward_clip,
            clip_range=reward_clip_range,
            scale=reward_scale,
        )
        rewards = torch.FloatTensor(reward_array)

        q_values = agent.q_network(states)
        if callable(scale_fn):
            q_values = scale_fn(q_values)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = agent.q_network(next_states)
            if callable(scale_fn):
                q_next = scale_fn(q_next)
            q_next_max = q_next.max(dim=1)[0]
        td = rewards + gamma * q_next_max - q_selected
        td_errors.append(float(td.abs().mean().item()))
        q_stds.append(float(q_values.std().item()))

    return {
        "td_errors": td_errors,
        "q_stds": q_stds,
        "avg_td": float(np.mean(td_errors)),
        "median_td": float(np.median(td_errors)),
        "avg_q_std": float(np.mean(q_stds)),
        "median_q_std": float(np.median(q_stds)),
    }


def sample_q_values(agent: Any, dataset_features: Any, sample_size: int, price_df: Any = None) -> np.ndarray:
    """
    Dataset üzerinden rastgele örnekler seçerek Q-değerlerini hesaplar.
    Environment kullanılarak gerçek 'Augmented State' oluşturulur.
    """
    total = len(dataset_features)
    if total == 0:
        return np.zeros((0, agent.q_network.network[-1].out_features))
        
    indices = np.random.choice(total, min(sample_size, total), replace=False)
    
    # Eğer price_df varsa Env kullanarak gerçek state üret (Tercih edilen)
    if price_df is not None:
        # Geçici Env oluştur
        env = RLTradingEnv(features_df=dataset_features, raw_df=price_df)
        states_list = []
        for idx in indices:
            # Env'in iç durumunu o indekse ayarla (Hackish ama etkili)
            env._current_step = idx
            # Rastgele bir pozisyon durumu ata ki çeşitlilik olsun
            env.position_fraction = random.choice([0.0, 0.5, 1.0])
            states_list.append(env._get_state())
        
        states = torch.FloatTensor(np.stack(states_list))
        
    else:
        # Fallback: Sadece feature'ları kullan (Padleme gerekebilir)
        raw_states = dataset_features.iloc[indices].to_numpy(dtype=float)
        if raw_states.shape[1] < agent.state_size:
            diff = agent.state_size - raw_states.shape[1]
            raw_states = np.pad(raw_states, ((0,0), (0, diff)), 'constant')
        states = torch.FloatTensor(raw_states)

    with torch.no_grad():
        q_values = agent.q_network(states)
        if hasattr(agent, "_scale_q"):
            q_values = agent._scale_q(q_values)
    return q_values.cpu().numpy()


def compute_q_histograms(
    q_values: np.ndarray,
    bins: int = 20,
    range_override: Optional[Tuple[float, float]] = None,
) -> Dict[int, Dict[str, Any]]:
    histograms: Dict[int, Dict[str, Any]] = {}
    num_actions = q_values.shape[1] if q_values.ndim == 2 else 0
    for action_idx in range(num_actions):
        data = q_values[:, action_idx]
        if range_override is not None:
            hist, bin_edges = np.histogram(data, bins=bins, range=range_override)
        else:
            hist, bin_edges = np.histogram(data, bins=bins)
        total = hist.sum()
        probabilities = (hist / total) if total > 0 else hist
        histograms[action_idx] = {
            "bins": bin_edges.tolist(),
            "counts": hist.tolist(),
            "probabilities": probabilities.tolist(),
        }
    return histograms


def inspect_checkpoint(
    model_path: Path,
    state_size: Optional[int] = None,
    dataset_path: Optional[Path] = None,
    replay_path: Optional[Path] = None,
    reward_clip: Optional[float] = None,
    reward_clip_range: Optional[Tuple[float, float]] = None,
    reward_scale: float = 1.0,
    batch_size: int = 32,
    samples: int = 10,
    histogram_bins: int = 20,
    histogram_sample: int = 1024,
    q_histogram: bool = False,
) -> Dict[str, Any]:
    
    dataset_features = None
    close_prices = None
    price_df = None
    env_state_size = None

    # 1. Dataset varsa yükle ve Env boyutunu öğren
    if dataset_path:
        logging.info(f"Loading dataset {dataset_path}...")
        dataset_features, price_df, _ = load_npz_dataset(dataset_path)
        close_prices = price_df["close"].to_numpy(dtype=float)
        
        # Env üzerinden state boyutunu öğren
        dummy_env = RLTradingEnv(features_df=dataset_features, raw_df=price_df)
        env_state_size = getattr(dummy_env, "state_dim", dataset_features.shape[1])
        logging.info(f"Detected Env State Size: {env_state_size}")

    # 2. State Size Belirle (Öncelik: CLI -> Env -> Checkpoint -> Varsayılan)
    if state_size is None:
        if env_state_size is not None:
            state_size = env_state_size
        elif dataset_features is not None:
            state_size = dataset_features.shape[1]
        else:
            state_size = 82 # En kötü durum fallback

    # 3. Ajanı Yükle
    agent = load_agent(model_path, state_size)

    # 4. Replay Buffer Hazırla
    if replay_path:
        replay = load_replay(replay_path)
    elif dataset_features is not None and close_prices is not None:
        replay = build_replay_from_dataset(dataset_features, close_prices)
    else:
        raise ValueError("Need either dataset_path or replay_path to evaluate checkpoint")

    if not replay:
        raise ValueError("Replay buffer is empty; cannot compute TD statistics")

    # 5. İstatistikleri Hesapla
    td_metrics = compute_td_stats(
        agent,
        replay,
        batch_size=batch_size,
        samples=samples,
        reward_clip=reward_clip,
        reward_clip_range=reward_clip_range,
        reward_scale=reward_scale,
    )
    reward_stats = compute_reward_stats(replay)

    histograms = None
    q_values_sample = None
    if q_histogram and dataset_features is not None:
        # Fiyat verisini de göndererek gerçekçi state (pozisyonlu) üretimi sağla
        q_values_sample = sample_q_values(agent, dataset_features, histogram_sample, price_df)
        histograms = compute_q_histograms(q_values_sample, bins=histogram_bins)

    return {
        "model_path": str(model_path),
        "state_size": state_size,
        "avg_td": td_metrics["avg_td"],
        "median_td": td_metrics["median_td"],
        "avg_q_std": td_metrics["avg_q_std"],
        "median_q_std": td_metrics["median_q_std"],
        "td_errors": td_metrics["td_errors"],
        "q_stds": td_metrics["q_stds"],
        "reward_stats": reward_stats,
        "histograms": histograms,
        "histogram_sample": histogram_sample,
        "histogram_bins": histogram_bins,
        "q_values_sample": q_values_sample,
        "samples": samples,
        "batch_size": batch_size,
    }


def print_inspection(summary: Dict[str, Any]) -> None:
    print("Checkpoint:", summary["model_path"])
    print(f"State size: {summary['state_size']}")
    print(f"Avg TD error: {summary['avg_td']:.6f}")
    print(f"Median TD error: {summary['median_td']:.6f}")
    print(f"Avg Q std: {summary['avg_q_std']:.6f}")
    print(f"Median Q std: {summary['median_q_std']:.6f}")
    reward_stats = summary["reward_stats"]
    print(
        "Reward stats (mean, std, min, max, 1%, 99%):",
        reward_stats["mean"],
        reward_stats["std"],
        reward_stats["min"],
        reward_stats["max"],
        reward_stats["1%"],
        reward_stats["99%"],
    )
    if summary["histograms"]:
        for action_idx, hist in summary["histograms"].items():
            print(f"Action {action_idx} histogram bins: {hist['bins']}")
            print(f"Action {action_idx} counts: {hist['counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RL checkpoints and replay statistics.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, help="Path to *.npz dataset for synthesizing replay and histogram sampling")
    parser.add_argument("--state-size", type=int, help="Force state size when instantiating agent")
    parser.add_argument(
        "--replay",
        type=Path,
        help="Optional replay pickle (trusted project artifact only)",
    )
    clip_group = parser.add_mutually_exclusive_group()
    clip_group.add_argument("--reward-clip", type=float, help="Symmetric reward clipping (±value)")
    clip_group.add_argument("--reward-clip-range", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--q-histogram", action="store_true")
    parser.add_argument("--histogram-bins", type=int, default=20)
    parser.add_argument("--histogram-sample", type=int, default=1024)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    
    configure_logging(args.log_level)

    clip_range = tuple(args.reward_clip_range) if args.reward_clip_range else None

    summary = inspect_checkpoint(
        model_path=args.model,
        state_size=args.state_size,
        dataset_path=args.dataset,
        replay_path=args.replay,
        reward_clip=args.reward_clip,
        reward_clip_range=clip_range,
        reward_scale=args.reward_scale,
        batch_size=args.batch_size,
        samples=args.samples,
        histogram_bins=args.histogram_bins,
        histogram_sample=args.histogram_sample,
        q_histogram=args.q_histogram,
    )

    print_inspection(summary)


if __name__ == "__main__":
    main()
