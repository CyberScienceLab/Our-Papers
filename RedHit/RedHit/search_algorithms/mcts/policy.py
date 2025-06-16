import random
import math
from typing import List
from RedHit.search_algorithms.mcts.node import Node


def random_policy(children):
    return random.choice(children)


def greedy_policy(children):
    return max(children, key=lambda c: c.reward / (c.visit + 1e-6))


def uct_policy(children: List[Node], epsilon=0.5, c=math.sqrt(2)):
    def uct_value(node: Node):
        if node.visit == 0:
            return float('inf')
        return (node.reward / node.visit) + c * math.sqrt(math.log(node.parent.visit) / node.visit)

    if random.random() < epsilon:
        return random.choice(children)

    return max(children, key=uct_value)
