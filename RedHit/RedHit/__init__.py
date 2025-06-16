from RedHit import *

import sys
import os


entry_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",))
sys.path.append(entry_dir)

# MS: control for later
#  from .search_algorithms import mcts
#  from .prompt_injection import  IndirectPromptGenerator
#  __all__ = ['mcts', 'IndirectPromptGenerator']
