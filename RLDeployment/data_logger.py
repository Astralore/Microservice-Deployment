# data_logger.py

import csv
import os
import numpy as np
from config import STATE_DIM, DATA_LOG_FILE

class DataLogger:
    def __init__(self, filename=DATA_LOG_FILE):
        self.filename = filename
        # Check if file exists to determine if header needs writing
        self.file_exists = os.path.isfile(filename)
        # Dynamically create header based on STATE_DIM
        self.header = [f'state_{i}' for i in range(STATE_DIM)] + ['action', 'q_value']
        print(f"DataLogger initialized for file: {self.filename}")
        if not self.file_exists:
             print("Log file does not exist, header will be written on first log.")

    def log(self, state, action, q_value, episode, step):
        """
        Logs a (State, Action, Q-value) tuple, possibly with episode/step info.
        Args:
            state (np.array or list): The state vector.
            action (int): The action taken.
            q_value (float): The predicted Q-value for the state-action pair.
            episode (int): Current episode number.
            step (int): Current step number within the episode.
        """
        # --- Filtering Logic (Optional) ---
        # Example: Only log data from later episodes or high-reward episodes
        # if episode < (MAX_EPISODES // 2): # Only log the second half of training
        #     return
        # Or, filter based on reward if reward is passed to log function
        # ----------------------------------

        try:
            # Convert state to list if it's a numpy array
            state_list = state.tolist() if isinstance(state, np.ndarray) else list(state)

            # Ensure data types are basic types for CSV writing
            row_data = state_list + [int(action), float(q_value)]

            # Include episode and step for context (optional)
            # row_data = [episode, step] + row_data
            # if using this, update the header accordingly

            with open(self.filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Write header only if the file was just created
                if not self.file_exists:
                    # Adjust header if episode/step are included
                    # writer.writerow(['episode', 'step'] + self.header)
                    writer.writerow(self.header)
                    self.file_exists = True # Mark that header is written

                writer.writerow(row_data)

        except IOError as e:
            print(f"Error writing to log file {self.filename}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during logging: {e}")