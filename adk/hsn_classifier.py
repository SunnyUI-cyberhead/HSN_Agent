# File: hsn_agent_adk/adk/hsn_classifier.py

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .exceptions import ModelLoadError, HSNPredictorError

class HSNClassifier:
    """
    Loads a pre‐trained Keras model, two tokenizers (for code and description),
    and a LabelEncoder. Exposes a .predict(...) method that returns a dict:
        {
            "predicted_hsn_code": <string>,
            "confidence": <float between 0 and 1>
        }
    """

    def __init__(
        self,
        model_path: str,
        code_tokenizer_path: str,
        desc_tokenizer_path: str,
        label_encoder_path: str,
        max_code_len: int,
        max_desc_len: int,
    ):
        try:
            # Load the Keras model
            self._model = load_model(model_path)
        except Exception as e:
            raise ModelLoadError(f"Failed to load Keras model from '{model_path}': {e}")

        # Load the code tokenizer
        try:
            with open(code_tokenizer_path, "rb") as f:
                self._code_tokenizer = pickle.load(f)
        except Exception as e:
            raise ModelLoadError(f"Failed to load code tokenizer from '{code_tokenizer_path}': {e}")

        # Load the description tokenizer
        try:
            with open(desc_tokenizer_path, "rb") as f:
                self._desc_tokenizer = pickle.load(f)
        except Exception as e:
            raise ModelLoadError(f"Failed to load description tokenizer from '{desc_tokenizer_path}': {e}")

        # Load the label encoder
        try:
            with open(label_encoder_path, "rb") as f:
                self._label_encoder = pickle.load(f)
        except Exception as e:
            raise ModelLoadError(f"Failed to load LabelEncoder from '{label_encoder_path}': {e}")

        # Store max lengths for padding/truncation
        self.max_code_len = max_code_len
        self.max_desc_len = max_desc_len

    def predict(self, hsn_code: str, description: str) -> dict:
        """
        Given a (partial) HSN code and a description string,
        return a dict with:
            {
              "predicted_hsn_code": <8-digit string>,
              "confidence": <float>
            }
        """
        try:
            # Preprocess the HSN code
            code_seq = self._code_tokenizer.texts_to_sequences([hsn_code])
            code_input = pad_sequences(
                code_seq,
                maxlen=self.max_code_len,
                padding="post",
                truncating="post",
            )

            # Preprocess the description
            desc_seq = self._desc_tokenizer.texts_to_sequences([description])
            desc_input = pad_sequences(
                desc_seq,
                maxlen=self.max_desc_len,
                padding="post",
                truncating="post",
            )

            # Run prediction
            probs = self._model.predict({"code_input": code_input, "desc_input": desc_input})
            idx = int(np.argmax(probs, axis=1)[0])
            predicted_code = self._label_encoder.inverse_transform([idx])[0]
            confidence = float(probs[0][idx])

            return {
                "predicted_hsn_code": str(predicted_code),
                "confidence": confidence
            }
        except Exception as e:
            raise HSNPredictorError(f"Error during HSN prediction: {e}")
