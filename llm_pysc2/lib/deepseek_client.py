# Copyright 2024, LLM-PySC2 Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DeepSeek API Client for LLM-PySC2.

Uses requests library directly to avoid openai==0.28 version conflicts.
DeepSeek API is OpenAI-compatible: https://api.deepseek.com/v1
"""

import requests
from loguru import logger
import threading
import random
import time


def deepseek_query_runtime(self):
    """Query runtime using requests for DeepSeek API (OpenAI-compatible)."""
    headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": self.model_name,
        "messages": self.messages,
        "temperature": self.temperature,
    }

    resp = requests.post(
        self.api_url,
        headers=headers,
        json=payload,
        timeout=self.config.MAX_LLM_RUNTIME_ERROR_TIME
    )
    resp.raise_for_status()
    data = resp.json()

    self.query_token_in = data.get("usage", {}).get("prompt_tokens", 0)
    self.query_token_out = data.get("usage", {}).get("completion_tokens", 0)
    self.llm_response = data["choices"][0]["message"]["content"]


class DeepSeekClient:
    """DeepSeek LLM client using requests library (OpenAI-compatible API).

    Avoids openai==0.28 version conflict by making HTTP calls directly.
    Maintains the same interface as GptClient for seamless integration.
    """

    def __init__(self, name, log_id, config):
        self.model_name = config.AGENTS[name]['llm']['model_name']
        self.api_base = config.AGENTS[name]['llm']['api_base']
        self.api_key = config.AGENTS[name]['llm']['api_key']
        self.temperature = config.temperature

        # Construct full API URL
        self.api_url = self.api_base.rstrip('/') + '/chat/completions'

        self.agent_name = name
        self.log_id = log_id
        self.config = config
        self.system_prompt = ''
        self.example_i_prompt = ''
        self.example_o_prompt = ''
        self.messages = []
        self.llm_response = None
        self.query_runtime = deepseek_query_runtime

        logger.info(f"[ID {self.log_id}] {self.agent_name} {self.model_name} DeepSeekClient initialized "
                     f"(api_url={self.api_url})")

        # Statistics (same fields as GptClient)
        self.num_query = 0
        self.query_time = 0
        self.query_token_in = 0
        self.query_token_out = 0
        self.total_query_time = 0
        self.total_query_token_in = 0
        self.total_query_token_out = 0
        self.ave_query_time = 0
        self.ave_query_token_in = 0
        self.ave_query_token_out = 0

    def wrap_message(self, obs_prompt, base64_image):
        """Assemble messages list in OpenAI-compatible format."""
        # DeepSeek currently does not support vision in v4-pro,
        # so we only use text mode regardless of base64_image
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.example_i_prompt},
            {"role": "assistant", "content": self.example_o_prompt},
            {"role": "user", "content": obs_prompt}
        ]

    def query(self, obs_prompt, base64_image=None):
        """Send query to DeepSeek API and return response text.

        Uses threading + timeout for robust error handling,
        same pattern as GptClient.query().
        """
        # Reset messages list
        self.wrap_message(obs_prompt, base64_image)

        # Send request with retry
        max_retries = self.config.MAX_LLM_QUERY_TIMES
        for retries in range(max_retries):
            try:
                logger.success(f"[ID {self.log_id}] {self.agent_name} Start calling DeepSeek API!")
                logger.debug(f"[ID {self.log_id}] {self.agent_name} input prompt: \n{obs_prompt}")

                self.thread = threading.Thread(target=self.query_runtime, args=(self,))
                self.thread.start()

                # Timeout handling
                query_start_time = float(time.time())
                while not isinstance(self.llm_response, str):
                    time.sleep(0.1)
                    if float(time.time()) - query_start_time > self.config.MAX_LLM_RUNTIME_ERROR_TIME:
                        logger.error(f"[ID {self.log_id}] {self.agent_name} DeepSeek query runtime error")
                        raise RuntimeError(f"{self.agent_name} DeepSeek query runtime error")

                if isinstance(self.llm_response, str):
                    self.num_query += 1
                    self.query_time = float(time.time()) - query_start_time
                    self.total_query_time += self.query_time
                    self.total_query_token_in += self.query_token_in
                    self.total_query_token_out += self.query_token_out
                    self.ave_query_time = self.total_query_time / self.num_query
                    self.ave_query_token_in = self.total_query_token_in / self.num_query
                    self.ave_query_token_out = self.total_query_token_out / self.num_query

                answer = self.llm_response
                logger.success(f"[ID {self.log_id}] {self.agent_name} Get DeepSeek response!")
                logger.debug(f"[ID {self.log_id}] {self.agent_name} DeepSeek response: \n{answer}")
                self.llm_response = None

                return answer

            except Exception as e:
                logger.error(f"[ID {self.log_id}] {self.agent_name} Error calling DeepSeek API: {e}")

                if retries >= max_retries - 1:
                    logger.error(
                        f"[ID {self.log_id}] {self.agent_name} Maximum retries reached. "
                        f"DeepSeek API is not responding."
                    )
                    return "I'm sorry, but I am unable to provide a response at this time due to technical difficulties."

                # Exponential backoff
                sleep_time = min((2 ** retries) + random.random(), 8 + random.random())
                logger.info(f"[ID {self.log_id}] {self.agent_name} Waiting {sleep_time}s before retry...")
                time.sleep(sleep_time)

        logger.error(f"[ID {self.log_id}] {self.agent_name} Can not get DeepSeek response after {max_retries} tries!")
        return f'[ID {self.log_id}] {self.agent_name} Can not get DeepSeek response after {max_retries} tries!'
