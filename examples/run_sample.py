"""

A minimal script to demonstrate HSNAgent in action.
Make sure to place your model and tokenizer artifacts in the same directory,
or adjust the paths accordingly.
"""

import os
import openai
from adk import HSNClassifier, HSNAgent, OpenAIAPIError, ModelLoadError, HSNPredictorError

def main():
    # 1. Ensure the API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY in your environment.")

    # 2. Instantiate the classifier:
    try:
        classifier = HSNClassifier(
            model_path="hsn_classifier_model.h5",
            code_tokenizer_path="code_tokenizer.pkl",
            desc_tokenizer_path="desc_tokenizer.pkl",
            label_encoder_path="label_encoder.pkl",
            max_code_len=10,
            max_desc_len=50
        )
    except ModelLoadError as e:
        print(f"Failed to load classifier: {e}")
        return

    # 3. Instantiate the agent
    agent = HSNAgent(classifier=classifier, openai_model="gpt-4o")

    # 4. Prompt details
    user_code = "1006.30"
    user_desc = "Rice, semi-milled or wholly milled, whether or not polished or glazed"

    # 5. Run and catch any errors
    try:
        response = agent.run(user_code, user_desc)
        print("\n=== AGENT FINAL RESPONSE ===")
        print(response)
    except (OpenAIAPIError, HSNPredictorError) as e:
        print(f"Error during agent run: {e}")

if __name__ == "__main__":
    main()
