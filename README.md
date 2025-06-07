# HSN Agent Development Kit

A Python “Agent Development Kit” (ADK) that:

1. Loads a trained Keras model + tokenizers + label encoder to predict an 8-digit HSN code from a partial code + description.  
2. Wraps OpenAI’s function-calling interface so GPT can decide when to call your local classifier.  


This is the trained HSN Model Link. 
https://github.com/SunnyUI-cyberhead/HSN_Agent/releases/download/h5/hsn_classifier_model.h5

# How It Works
adk/hsn_classifier.py
Loads your .h5 Keras model, the two Tokenizer pickles, and the LabelEncoder to expose .predict(hsn_code, description) → {predicted_hsn_code, confidence}.

adk/openai_agent.py
Uses openai.ChatCompletion with functions=[{…hsn_classifier spec…}] so GPT-4 (or 3.5-turbo) can choose to call your classifier or reply directly.

adk/utils.py
Retry logic, logging, and a safe wrapper around the OpenAI API.

examples/run_sample.py
Glues everything together:

Loads constants.pkl to get max_code_len and max_desc_len

Instantiates HSNClassifier(...)

Instantiates HSNAgent(...)

Calls agent.run("1006.30", "Rice…")
