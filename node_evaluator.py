import os
from typing import Type
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
import inspect

from prompts.meditation import TRANSCRIPT_PROMPT
from prompts.conversation import CONVERSATION_PROMPT

load_dotenv(override=True)

# ---------------------------------------------------------
# Pydantic Models for Judge Outputs (Strict Rubrics)
# ---------------------------------------------------------


class GeneratorScore(BaseModel):
    creativity_index: int = Field(
        description="Scale 1-5: Vividness and flow for spoken audio.")
    format_compliance: int = Field(
        description="Binary 0/1: Is it valid Markdown with breathing pause indicators like [...]?")
    length_penalty: int = Field(
        description="Binary 0/1: Is it strictly under 100 words? 1 if yes, 0 if it failed.")
    reasoning: str = Field(description="Brief justification for the scores.")


class ConversationScore(BaseModel):
    empathy_score: int = Field(
        description="Scale 1-5: Did the agent validate the user's feelings?")
    preference_capture: int = Field(
        description="Count 0-3: How many traits (mood, goal, environment) were extracted?")
    reasoning: str = Field(description="Brief justification.")


class OrchestrationScore(BaseModel):
    state_awareness: int = Field(
        description="Binary 0/1: Did the agent correctly identify the user is ready for generation?")
    reasoning: str = Field(description="Brief justification.")

# ---------------------------------------------------------
# 2. Mocked Prompts & Guardrails
# ---------------------------------------------------------


TEST_SUITE = {
    "generator": {
        "system": TRANSCRIPT_PROMPT,
        "input": "I am feeling very anxious about an upcoming presentation. I need to ground myself.",
        "schema": GeneratorScore
    },
    "conversation": {
        "system": CONVERSATION_PROMPT,
        "input": "User: Hi, I just got home from a really long shift. I can't seem to unwind.",
        "schema": ConversationScore
    },
    "orchestration": {
        "system": "You are a routing agent. Review the conversation history. If you know the user's mood AND goal, output JSON: {'proceed': true}. Otherwise, {'proceed': false}.",
        "input": "History: \nUser: I'm stressed (mood).\nAgent: I hear you. What would you like to achieve today?\nUser: I just want to fall asleep easily (goal).",
        "schema": OrchestrationScore
    }
}

# ---------------------------------------------------------
# 3. The Evaluation Engine
# ---------------------------------------------------------


class NodeEvaluationSuite:
    def __init__(self, hf_endpoint_url: str):
        # The model being tested (Creative)

        hf_token = os.getenv("HF_TOKEN")

        self.target_llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id=hf_endpoint_url,
                task="text-generation",
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                max_new_tokens=256,
                huggingfacehub_api_token=hf_token,
                provider="auto"
            )
        )

        # The Judge model (Strict, Deterministic)
        self.judge_llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id=hf_endpoint_url,
                task="text-generation",
                temperature=0.0,
                max_new_tokens=150,
                top_p=0.5,
                top_k=20,
                huggingfacehub_api_token=hf_token,
                provider="auto"
            )
        )

    def _generate_candidate(self, system_prompt: str, user_input: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]
        ai_message = self.target_llm.invoke(messages)
        return ai_message.content

    def _judge_candidate(self, node_type: str, candidate_output: str, schema_class: Type[BaseModel]) -> Type[BaseModel]:
        # Ensure the judge is strictly bound to the Pydantic class
        judge_system_prompt = inspect.cleandoc("""
        You are an impartial judge. 
        You are validating the provided text.
        The final judgement output must be in the following format:
  
        {format_instructions}
                                               
        IMPORTANT: Output ONLY well structured JSON. Do NOT output anything else.
        """)

        parser = PydanticOutputParser(pydantic_object=schema_class)

        system_message_template = SystemMessagePromptTemplate.from_template(
            judge_system_prompt, partial_variables={"format_instructions": parser.get_format_instructions()})

        candidate_prompt = inspect.cleandoc("""Text to validate:
        {candidate_output}
        """)

        candidate_prompt_template = HumanMessagePromptTemplate.from_template(
            candidate_prompt)

        judge_prompt = ChatPromptTemplate.from_messages(
            [system_message_template, candidate_prompt_template])

        return (judge_prompt | self.judge_llm | parser).invoke(
            {"candidate_output": candidate_output})

    def run_node_test(self, node_type: str, n_iterations: int = 100) -> Dict[str, Any]:
        print(
            f"\n--- Testing Node: {node_type.upper()} ({n_iterations} iterations) ---")

        config = TEST_SUITE[node_type]
        results = []

        for _ in tqdm(range(n_iterations), desc="Generating & Judging"):
            candidate = self._generate_candidate(
                config["system"], config["input"])

            score = self._judge_candidate(
                node_type, candidate, config["schema"])

            results.append(score.model_dump())

        return self._aggregate_results(results)

    def _aggregate_results(self, results: List[Dict]) -> Dict[str, float]:
        if not results:
            return {"error": "All judge evaluations failed to parse."}

        aggregated = {}
        # Calculate mean for all numeric keys
        keys = [k for k in results[0].keys() if isinstance(
            results[0][k], (int, float))]

        for key in keys:
            values = [r[key] for r in results]
            aggregated[f"{key}_mean"] = round(np.mean(values), 2)
            aggregated[f"{key}_variance"] = round(np.var(values), 2)

        return aggregated


# ---------------------------------------------------------
# 4. Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Note: Replace with your actual local or hosted HF model ID/Endpoint
    # hf_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    hf_model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

    suite = NodeEvaluationSuite(hf_endpoint_url=hf_model_id)

    # Run a smaller batch for demonstration (change to 100 for full test)
    for rubric in TEST_SUITE:
        generator_metrics = suite.run_node_test(rubric, n_iterations=10)
        print(f"\n{rubric.title()} Node Metrics:")
        for k, v in generator_metrics.items():
            print(f"{k}: {v}")
