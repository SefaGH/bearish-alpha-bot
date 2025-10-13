"""
Experience Replay System for Reinforcement Learning.

Implements prioritized experience replay for efficient learning from
trading experiences with priority sampling based on TD-error.
"""

import numpy as np
from collections import deque
import random
import logging
from typing import List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class ExperienceReplay:
    """Advanced experience replay buffer with priority sampling."""
    
    def __init__(self, max_size: int = 100000, priority_alpha: float = 0.6):
        """
        Initialize experience replay buffer.
        
        Args:
            max_size: Maximum size of replay buffer
            priority_alpha: Priority exponent (0 = uniform, 1 = full priority)
        """
        self.max_size = max_size
        self.priority_alpha = priority_alpha
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.max_priority = 1.0
        
        # Statistics
        self.total_experiences = 0
        self.sample_count = 0
        
        logger.info(f"Initialized ExperienceReplay buffer with max_size={max_size}, priority_alpha={priority_alpha}")
    
    def add_experience(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool, td_error: Optional[float] = None):
        """
        Add trading experience to replay buffer.
        
        Args:
            state: State at time t
            action: Action taken
            reward: Reward received
            next_state: State at time t+1
            done: Whether episode ended
            td_error: Temporal difference error (for priority calculation)
        """
        # Store experience tuple
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
        # Calculate priority based on TD error
        if td_error is not None:
            priority = (abs(td_error) + 1e-6) ** self.priority_alpha
        else:
            # Use max priority for new experiences
            priority = self.max_priority
        
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
        
        self.total_experiences += 1
        
        if self.total_experiences % 1000 == 0:
            logger.debug(f"Added experience #{self.total_experiences}, buffer size: {len(self.buffer)}")
    
    def sample_batch(self, batch_size: int, beta: float = 0.4) -> List[Tuple]:
        """
        Sample batch of experiences using prioritized sampling.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (increases to 1 during training)
            
        Returns:
            List of experience tuples
        """
        if len(self.buffer) < batch_size:
            logger.warning(f"Not enough experiences: {len(self.buffer)} < {batch_size}")
            return []
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
            p=probabilities
        )
        
        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        
        self.sample_count += 1
        
        return batch
    
    def sample_uniform(self, batch_size: int) -> List[Tuple]:
        """
        Sample batch uniformly (without priority).
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of experience tuples
        """
        if len(self.buffer) < batch_size:
            return []
        
        return random.sample(list(self.buffer), batch_size)
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of experiences to update
            td_errors: New TD errors for priority calculation
        """
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.priorities):
                priority = (abs(td_error) + 1e-6) ** self.priority_alpha
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def get_recent_experiences(self, n: int = 10) -> List[Tuple]:
        """
        Get most recent experiences.
        
        Args:
            n: Number of recent experiences to return
            
        Returns:
            List of recent experience tuples
        """
        if len(self.buffer) == 0:
            return []
        
        n = min(n, len(self.buffer))
        return list(self.buffer)[-n:]
    
    def get_high_priority_experiences(self, n: int = 10) -> List[Tuple]:
        """
        Get experiences with highest priorities.
        
        Args:
            n: Number of high-priority experiences to return
            
        Returns:
            List of high-priority experience tuples
        """
        if len(self.buffer) == 0:
            return []
        
        # Get indices sorted by priority
        priorities = np.array(self.priorities)
        top_indices = np.argsort(priorities)[-n:]
        
        return [self.buffer[idx] for idx in top_indices]
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()
        self.priorities.clear()
        self.max_priority = 1.0
        logger.info("Experience replay buffer cleared")
    
    def get_statistics(self) -> dict:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary with buffer statistics
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'max_size': self.max_size,
                'total_experiences': self.total_experiences,
                'sample_count': self.sample_count,
                'utilization': 0.0
            }
        
        priorities = np.array(self.priorities)
        
        return {
            'size': len(self.buffer),
            'max_size': self.max_size,
            'total_experiences': self.total_experiences,
            'sample_count': self.sample_count,
            'utilization': len(self.buffer) / self.max_size,
            'priority_stats': {
                'mean': float(priorities.mean()),
                'std': float(priorities.std()),
                'min': float(priorities.min()),
                'max': float(priorities.max())
            }
        }
    
    def save_buffer(self, path: str):
        """
        Save buffer to disk.
        
        Args:
            path: File path to save buffer
        """
        import pickle
        
        data = {
            'buffer': list(self.buffer),
            'priorities': list(self.priorities),
            'max_priority': self.max_priority,
            'total_experiences': self.total_experiences,
            'sample_count': self.sample_count
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Experience replay buffer saved to {path}")
    
    def load_buffer(self, path: str):
        """
        Load buffer from disk.
        
        Args:
            path: File path to load buffer from
        """
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.buffer = deque(data['buffer'], maxlen=self.max_size)
        self.priorities = deque(data['priorities'], maxlen=self.max_size)
        self.max_priority = data['max_priority']
        self.total_experiences = data['total_experiences']
        self.sample_count = data['sample_count']
        
        logger.info(f"Experience replay buffer loaded from {path} with {len(self.buffer)} experiences")
    
    def __len__(self):
        """Return buffer size."""
        return len(self.buffer)
    
    def __repr__(self):
        """String representation."""
        return f"ExperienceReplay(size={len(self.buffer)}/{self.max_size}, priority_alpha={self.priority_alpha})"


class EpisodeBuffer:
    """Buffer for storing complete episodes of trading experiences."""
    
    def __init__(self, max_episodes: int = 1000):
        """
        Initialize episode buffer.
        
        Args:
            max_episodes: Maximum number of episodes to store
        """
        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)
        self.current_episode = []
        
        logger.info(f"Initialized EpisodeBuffer with max_episodes={max_episodes}")
    
    def add_transition(self, state: np.ndarray, action: int, reward: float,
                      next_state: np.ndarray, done: bool):
        """
        Add transition to current episode.
        
        Args:
            state: State at time t
            action: Action taken
            reward: Reward received
            next_state: State at time t+1
            done: Whether episode ended
        """
        transition = (state, action, reward, next_state, done)
        self.current_episode.append(transition)
        
        if done:
            self.end_episode()
    
    def end_episode(self):
        """End current episode and store it."""
        if self.current_episode:
            episode_return = sum(t[2] for t in self.current_episode)
            episode_data = {
                'transitions': self.current_episode,
                'length': len(self.current_episode),
                'return': episode_return
            }
            self.episodes.append(episode_data)
            logger.debug(f"Episode ended: length={len(self.current_episode)}, return={episode_return:.4f}")
            self.current_episode = []
    
    def get_best_episodes(self, n: int = 10) -> List[dict]:
        """
        Get episodes with highest returns.
        
        Args:
            n: Number of best episodes to return
            
        Returns:
            List of best episode dictionaries
        """
        if not self.episodes:
            return []
        
        sorted_episodes = sorted(self.episodes, key=lambda x: x['return'], reverse=True)
        return sorted_episodes[:n]
    
    def get_recent_episodes(self, n: int = 10) -> List[dict]:
        """
        Get most recent episodes.
        
        Args:
            n: Number of recent episodes to return
            
        Returns:
            List of recent episode dictionaries
        """
        if not self.episodes:
            return []
        
        n = min(n, len(self.episodes))
        return list(self.episodes)[-n:]
    
    def get_statistics(self) -> dict:
        """
        Get episode statistics.
        
        Returns:
            Dictionary with episode statistics
        """
        if not self.episodes:
            return {
                'num_episodes': 0,
                'avg_length': 0,
                'avg_return': 0
            }
        
        lengths = [ep['length'] for ep in self.episodes]
        returns = [ep['return'] for ep in self.episodes]
        
        return {
            'num_episodes': len(self.episodes),
            'avg_length': np.mean(lengths),
            'avg_return': np.mean(returns),
            'best_return': np.max(returns),
            'worst_return': np.min(returns),
            'std_return': np.std(returns)
        }
    
    def clear(self):
        """Clear all episodes."""
        self.episodes.clear()
        self.current_episode = []
        logger.info("Episode buffer cleared")
    
    def __len__(self):
        """Return number of stored episodes."""
        return len(self.episodes)
    
    def __repr__(self):
        """String representation."""
        return f"EpisodeBuffer(episodes={len(self.episodes)}/{self.max_episodes})"
