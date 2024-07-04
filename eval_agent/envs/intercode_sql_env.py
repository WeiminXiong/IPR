import re
import json
import logging
from typing import Any, Dict, List, Tuple

from eval_agent.envs import BaseEnv
from eval_agent.tasks import IntercodeSQLTask
from eval_agent.prompt import prompt_with_icl
from eval_agent.utils.datatypes import State


logger = logging.getLogger("agent_frame")


class IntercodeSQLEnv(BaseEnv):
    def __init__(
        self,
        task: IntercodeSQLTask,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: IntercodeSQLTask = task
        self.env = task.env
        self.state = State()
    
    def parse_conduct_action(self, llm_output: str) -> str:
        done = False
        reward = 0
        llm_output = llm_output.strip()
        try:
            llm_output = llm_output.strip()
            action_pattern = re.compile(r"Action:\s?(.*)", re.DOTALL)
            action = re.findall(action_pattern, llm_output)[0]
            assert action is not None
            if "submit" in action:
                observation, reward, done, info = self.env.step("submit")
                return observation, reward, done
            else:
                code_part = [
                        i.strip()
                        for i in re.findall(
                            r"```sql(.*?)```", action, re.DOTALL
                        )
                    ][0]
                code_part = code_part.split("\n")[0]

                if code_part== "":
                    raise Exception("No output")
                output, reward, done, info = self.env.step(code_part)
                if "Error" in output and not "Unknown" in output:
                    observation = f"{output}\n"
                else:
                    if output == "":
                        observation = "[Executed Successfully with No Output]"
                    else:
                        observation = f"{output}"
                if isinstance(observation, str) and len(observation) >350:
                    observation = observation[:350] + "..."
                return observation, reward, done
        except Exception as e:
            observation = "I don't understand your input.\n Your input shoud include the key world \"Action:\".\n If you want to perform a query operation, please use the following format:\n Action: \n```sql\nthe mysql code\n```. And the action should be executable and should not contain natural language.\n If you have got the final answer, please use the following format:\n Action: submit. Remember your output should contain only one \"Action:\" part."
        
        return observation, reward, done
    
    
    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        
        observation, reward, done = self.parse_conduct_action(llm_output)

        observation = f"Observation: {observation}"
        self.state.history.append({
            "role": "user",
            "content": observation,
        })

        self.state.steps += 1
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = 0

        if done:
            if reward == 1.0:
                self.state.finished = True
                self.state.success = True
                self.state.terminate_reason = "success"
                self.state.reward = reward
            else:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "Error answer."
                self.state.reward = reward

        return observation, self.state

    def reset(self) -> Tuple[str, State]:
        self.state = State()
        self.env.reset(self.task.task_id)
        cur_task = self.env.query
        observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, 1)
        if self.icl_format == 'first':
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
        elif self.icl_format == 'conversation':
            self.state.history = messages
        return observation, self.state
