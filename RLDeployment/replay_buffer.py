# replay_buffer.py

from collections import deque
import random
import numpy as np

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity):
        """Initialize a ReplayBuffer object."""
        self.buffer = deque(maxlen=capacity)

    def store(self, experience):
        """Add a new experience to memory."""
        # experience should be (state, action, reward, next_state, done, mask, next_mask)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def update_last_reward(self, final_reward):
        """
        Updates the reward of the most recently added transition.
        Call this after an episode finishes and the final reward is known.
        """
        if not self.buffer:
            return

        # Get the last experience, which should correspond to the last step
        last_experience = list(self.buffer[-1]) # Convert tuple to list for modification

        # Combine immediate reward (if any) with final reward.
        # Option 1: Add final reward to the last step's immediate reward
        last_experience[2] += final_reward
        # Option 2: Replace the last step's reward with the final reward (if no meaningful immediate reward)
        # last_experience[2] = final_reward

        # Store the updated experience back
        self.buffer[-1] = tuple(last_experience)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)