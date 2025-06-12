# File: meta_agent_env.py
import os
import sys
import pandas as pd
import numpy as np
import gym
from gym import spaces

# === Add project root to path ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)