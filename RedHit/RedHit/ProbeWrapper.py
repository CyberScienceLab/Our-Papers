
from config import RedHitConf
from global_result import GlobalResult
from prompt_injection.prompt_generator import IndirectPromptGenerator
from search_algorithms.mcts.mc_tree_sarch import MCTreeSearch
import copy
from evaluator import Evaluator


class ProbeWrapper():
    prompt_generator = None

    def __init__(self, probe):
        self.probe = probe
        model_name = RedHitConf.get('default_model_path')
        self.probe.detectors = []
        self.probe.evaluator = None
        if ProbeWrapper.prompt_generator is None:
            ProbeWrapper.prompt_generator = IndirectPromptGenerator(model_name)

        self.prompt_generator = ProbeWrapper.prompt_generator

    def _execute_attempt(self, this_attempt):
        """handles sending an attempt to the generator, postprocessing, and logging"""

        self.probe._generator_precall_hook(
            self.probe.generator, this_attempt)

        iteration_count = 100
        max_depth = 7
        breadth_node_count = 6
        final_depth = 15
        reward_treshold = 0.8

        evaluator = Evaluator(self.probe)
        copy_attempt = copy.deepcopy(this_attempt)

        mcts = MCTreeSearch(
            iteration_count=iteration_count,
            base_prompt=copy_attempt.prompt,
            max_depth=max_depth,
            breadth_node_count=breadth_node_count,
            prompt_generator=self.prompt_generator,
            target_LLM=self.probe.generator.generate,
            evaluator=evaluator,
            attempt=copy_attempt,
            garak_generations=self.probe.generations,
            reward_treshold=reward_treshold,
            final_depth=final_depth
        )

        mcts.start()

        result = mcts.results

        with open('/home/researchuser/LLMSec/LLMSecurity/RedHit/mistral_result.txt', 'w') as file:
            for integer in result:
                file.write(str(integer) + '\n')

        # this_attempt.outputs = self.probe.generator.generate(
        #     this_attempt.prompt, generations_this_call=self.probe.generations)

        # if self.probe.post_buff_hook:
        #     this_attempt = self.probe._postprocess_buff(
        #         this_attempt)

        # this_attempt = self.probe._postprocess_hook(this_attempt)
        # self.probe._generator_cleanup()

        # return copy.deepcopy(this_attempt)

    # def _execute_attempt_fake(self, this_attempt):
    #     """handles sending an attempt to the generator, postprocessing, and logging"""

    #     dspy_program  = self.load_dspy_program()

    #     self.probe_object_self._generator_precall_hook(self.probe_object_self.generator, this_attempt)

    #     prompt_count= 10
    #     passed_prompts=[]
    #     training_epoch = 5
    #     minimum_data_size= 20
    #     trainset= []
    #     deceptive_prompt=""
    #     tricked_output=""
    #     all_test_passed= True
    #     epoch_accuracies=[]
    #     #Mohsen:
    #     # Bootstrapping& red-teaming
    #     #for epoch in range(training_epoch):

    #     base_prompt= this_attempt.prompt
    #     indirect_prompts= dspy_program(base_prompt= base_prompt, prompt_count=prompt_count)
    #     indirect_prompts.append(base_prompt)
    #     epoch_score= []
    #     for prompt in indirect_prompts:
    #         current_attempt_score=0
    #         test_passed= True
    #         test_attempt= copy.deepcopy(this_attempt)
    #         test_attempt.prompt= prompt
    #         test_attempt.outputs = self.probe_object_self.generator.generate(test_attempt.prompt, generations_this_call=self.probe_object_self.generations )

    #         # for d in self.probe_object_self.detectors:
    #         #     detections= d.detect(test_attempt)
    #         #     for r in map(float,  detections):
    #         #         if(not self.probe_object_self.evaluator.test(r)):
    #         #             test_passed= False
    #         #             all_test_passed= False
    #         #             current_attempt_score+= 1
    #         #             break

    #         #     if not test_passed:
    #         #         deceptive_prompt=  prompt
    #         #         tricked_output= test_attempt.outputs
    #         #         break

    #         epoch_score.append((current_attempt_score / (len(self.probe_object_self.detectors)+ len(indirect_prompts))) *100)
    #         if not test_passed:
    #             passed_prompts.append(dspy.Example( question=  base_prompt, response=prompt).with_inputs('question'))

    #     avg_acc= sum(epoch_score) / len(epoch_score)
    #     epoch_accuracies.append(avg_acc)
    #     print('*****************************' + self.probe_object_self.probe_name, avg_acc)

    #         # Mohsen:
    #         # Optimization
    #         if len(passed_prompts) >= minimum_data_size:
    #             dspy_program.optimize(passed_prompts)
    #             trainset+= passed_prompts
    #             passed_prompts= []

    #     if not all_test_passed:
    #         this_attempt.prompt = deceptive_prompt
    #         this_attempt.output= tricked_output

    #     self.save_dspy_program(dspy_program)

    #     if self.probe_object_self.post_buff_hook:
    #         this_attempt = self.probe_object_self._postprocess_buff(this_attempt)

    #     this_attempt = self.probe_object_self._postprocess_hook(this_attempt)
    #     self.probe_object_self._generator_cleanup()

    #     return copy.deepcopy(this_attempt)
