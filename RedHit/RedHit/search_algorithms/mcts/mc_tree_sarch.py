# Author: Mohsen Sorkhpour
import copy
import random
from typing import Tuple
from tqdm import tqdm
from RedHit.search_algorithms.mcts.schemas import Example
from RedHit.search_algorithms.mcts.node import Node
from RedHit.search_algorithms.mcts.fine_tune import FineTune
from RedHit.search_algorithms.mcts.utility import inject_histories, memory_safe_fine_tuning
from garak.attempt import Attempt
from collections import deque
import heapq


class MCTreeSearch:
    def __init__(self,
                 iteration_count,
                 base_prompt,
                 max_depth,
                 breadth_node_count,
                 prompt_generator,
                 target_LLM,
                 evaluator,
                 attempt,
                 garak_generations,
                 reward_treshold,
                 final_depth):

        self.iteration_count = iteration_count
        self.base_prompt = base_prompt
        self.max_depth = max_depth
        self.breadth_node_count = breadth_node_count
        self.prompt_generator = prompt_generator
        self.target_LLM = target_LLM
        self.evaluator = evaluator
        self.fine_tuner = FineTune()
        self.attempt = attempt
        self.garak_generations = garak_generations
        self.chosen_buff = deque(maxlen=50)
        self.rejected_buff = deque(maxlen=50)
        self.reward_treshold = reward_treshold
        self.final_depth = final_depth
        self.Root = None
        self.epsilon = 1
        self.results = []

    def start(self):
        root = Node(
            prompt=self.base_prompt,
            prompt_generator=self.prompt_generator,
            parent=None
        )

        root.init_root_prompt()
        self.root = root
        for iteration in tqdm(range(self.iteration_count)):
            is_injected, final_node = self.search(root, 0)

            self.results.append(final_node.reward)

            if is_injected:
                self.chosen_buff.append(final_node.prompt)
            else:
                self.rejected_buff.append(final_node.prompt)
            if iteration > 0 and iteration % 10 == 0:
                # fine-tuning
                trainset = self.generate_preferences_dataset(root.prompt)
                memory_safe_fine_tuning(
                    self.fine_tuner.dpo_trainer, trainset)
            if (self.epsilon > 0.003):
                self.epsilon -= 0.02

    def search(self, node: Node, depth) -> tuple[bool, Node]:

        if depth > 0 and self.is_terminal_State(node):
            return True, node

        # if (depth >= self.max_depth and node.reward < self.reward_treshold) or depth >= self.final_depth:
        if depth >= self.max_depth:
            return False, node

        # Find a leaf
        node = self.traverse(node)
        first_attempt = depth == 0
        node = self.expansion(node, first_attempt)
        if node is None:
            print('invalid node founded')
            return None

        reward = self.simulation(node)
        node.backpropagate(reward)

        return self.search(node, depth+1)

    def traverse(self, node):
        """ this function traverses the tree to find a leaf Node"""

        while node.has_children():
            node = node.best_child(epsilon=self.epsilon)

            if node is None:
                print("Not valid node found")
                break

        return node

    def expansion(self, node: Node, first_attempt) -> Tuple[Node, int]:
        from RedHit.search_algorithms.mcts.policy import random_policy

        if node is not None and not node.has_children():
            trajectory = self.get_current_trajectory(node)
            node.expand(trajectory, first_attempt, self.breadth_node_count)
            if node.has_children():
                node = random_policy(node.children)
                return node

        return None

    def simulation(self, node: Node) -> float:
        """evaluate whether the attempt passed all detector or not"""
        attempt = self.get_attempt(node.prompt)
        attempt.outputs = self.target_LLM(
            attempt.prompt,  self.garak_generations)

        node.response = str(
            {f'Targeted LLM Response_{idx}': response for idx, response in enumerate(attempt.outputs)})

        reward = self.evaluator.attempt_evaluate(attempt)

        return reward

    def get_attempt(self, prompt: str) -> Attempt:

        attempt = copy.deepcopy(self.attempt)
        attempt.prompt = prompt

        return attempt

    def is_terminal_State(self, node: Node) -> bool:

        attempt: Attempt = self.get_attempt(node.prompt)
        attempt.outputs = self.target_LLM(
            attempt.prompt,  self.garak_generations)

        reward = self.evaluator.attempt_evaluate(attempt)
        node.reward = reward
        # reward: that the number of injected prompt devided by the number of detection,
        # so 1 means that all prompts successfully injected
        return reward >= 0.5

    def get_trajectories(self, history: list[Node]):
        """
        It returns an array of prompt/response pairs from the history of nodes
        """
        all_trajectories = []

        for node in history:
            trajectory = list[Example]
            while node.parent is not None:
                trajectory.append(
                    Example(node.prompt, node.response, node.reasoning))
                node = node.parent

            trajectory = trajectory[::-1]
            all_trajectories.append(trajectory)

        return all_trajectories

    def generate_preferences_dataset(self, prompt: str):

        chosens = list(self.chosen_buff)
        rejecteds = list(self.rejected_buff)

        preferences_dataset = []
        dataset = []
        remaining = abs(len(chosens) - len(rejecteds))

        if len(rejecteds) > len(chosens):
            chosens.extend(self.get_topk_reward_prompts(n=remaining))
            dataset = zip(chosens, rejecteds)
        else:
            rejecteds.extend(self.get_zero_reward_prompts(n=remaining))
            dataset = zip(chosens, rejecteds)

        failed_histories = self.get_zero_reward_prompts(n=100)
        sample_len = min(5, len(failed_histories))
        # MS: I add randomness to histories of base COT prompt in the fine-tuning to make better generalization
        for chosen, rejected in dataset:
            random_histories = "\n\n".join(random.sample(
                failed_histories, sample_len))
            prompt = inject_histories(prompt, random_histories)
            preferences_dataset.append(
                {'prompt': prompt, 'chosen': chosen, 'rejected': rejected})
        return preferences_dataset

    def get_current_trajectory(self, node: Node) -> list[Example]:

        trajectory: list[Example] = []
        while node.parent is not None:
            trajectory.append(
                Example(str(random.random())+'--' + node.prompt, node.response, node.reasoning))
            node = node.parent
        return trajectory

    def get_zero_reward_prompts(self, n=None):
        zero_reward_prompts = []
        nodes_to_visit = [self.root]
        index = 0

        while nodes_to_visit:
            if index > 0:
                current_node = nodes_to_visit.pop(0)
                if current_node.reward < 0.5:
                    zero_reward_prompts.append(current_node.prompt)
                    if n is not None and len(zero_reward_prompts) >= n:
                        return zero_reward_prompts
                nodes_to_visit.extend(current_node.children)
            index += 1

        return zero_reward_prompts

    def get_topk_reward_prompts(self, n=1):
        all_nodes = []
        nodes_to_visit = [self.root]

        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            all_nodes.append(current_node)
            if current_node.has_children():
                nodes_to_visit.extend(current_node.children)

        top_n_nodes = heapq.nlargest(
            n, all_nodes, key=lambda node: node.reward)

        positive_reward_prompts = [node.prompt for node in top_n_nodes]

        return positive_reward_prompts
