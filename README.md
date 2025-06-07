# HSN Agent Development Kit

A Python “Agent Development Kit” (ADK) that:

1. Loads a trained Keras model + tokenizers + label encoder to predict an 8-digit HSN code from a partial code + description.  
2. Wraps OpenAI’s function-calling interface so GPT can decide when to call your local classifier.  


This is the trained HSN Model Link. 
https://github.com/SunnyUI-cyberhead/HSN_Agent/releases/download/h5/hsn_classifier_model.h5
