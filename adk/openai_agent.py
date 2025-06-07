# File: hsn_agent_adk/adk/openai_agent.py

import json
import openai
from .utils import safe_chat_completion, validate_api_key, logger
from .hsn_classifier import HSNClassifier
from .exceptions import OpenAIAPIError, HSNPredictorError

class HSNAgent:
    """
    Wraps OpenAI function‐calling for "hsn_classifier". On initialization,
    pass in a fully‐configured HSNClassifier instance. Then call .run(user_code, user_desc)
    to get the agent’s final response string.
    """

    def __init__(self, classifier: HSNClassifier, openai_model: str = "gpt-4o"):
        validate_api_key()
        self.classifier = classifier
        self.model = openai_model

        # Define the function schema exactly as in your notebook
        self.function_spec = {
            "name": "hsn_classifier",
            "description": "Given an HSN code and description, return the predicted 8-digit HSN code and confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hsn_code": {
                        "type": "string",
                        "description": "The raw HSN code to validate"
                    },
                    "description": {
                        "type": "string",
                        "description": "Text description of the item"
                    }
                },
                "required": ["hsn_code", "description"],
            }
        }

    def run(self, user_code: str, user_desc: str) -> str:
        """
        1. Send an initial message to GPT asking it to verify/correct the HSN code.
        2. If GPT requests a function call (hsn_classifier), run it locally and resend.
        3. Return GPT’s final answer string.
        """
        try:
            # 1) Initial prompt
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"I have HSN code '{user_code}' and description '{user_desc}'. "
                        "Please verify or correct the HSN code and tell me how confident you are."
                    )
                }
            ]

            # 2) Let GPT decide if it wants to call hsn_classifier
            response1 = safe_chat_completion(
                model=self.model,
                messages=messages,
                functions=[self.function_spec],
                function_call="auto"
            )
            message = response1["choices"][0]["message"]

            # 3) If GPT asked for the function
            if message.get("function_call") is not None:
                func_name = message["function_call"]["name"]
                try:
                    func_args = json.loads(message["function_call"]["arguments"])
                except json.JSONDecodeError as e:
                    raise OpenAIAPIError(f"Failed to decode function arguments: {e}")

                if func_name == "hsn_classifier":
                    # Call our local classifier
                    try:
                        result = self.classifier.predict(
                            hsn_code=func_args.get("hsn_code", ""),
                            description=func_args.get("description", "")
                        )
                    except HSNPredictorError as e:
                        raise

                    # 4) Send GPT the function output
                    second_messages = [
                        # Resend the original user prompt
                        messages[0],
                        {
                            "role": "assistant",
                            "content": None,
                            "function_call": {
                                "name": func_name,
                                "arguments": json.dumps(func_args)
                            }
                        },
                        {
                            "role": "function",
                            "name": func_name,
                            "content": json.dumps(result)
                        }
                    ]
                    response2 = safe_chat_completion(
                        model=self.model,
                        messages=second_messages
                    )
                    final_message = response2["choices"][0]["message"]["content"]
                    return final_message.strip()
                else:
                    raise OpenAIAPIError(f"GPT requested unknown function: {func_name}")
            else:
                # GPT answered directly without calling our function
                return message["content"].strip()

        except Exception as e:
            logger.exception("Error in HSNAgent.run()")
            raise
