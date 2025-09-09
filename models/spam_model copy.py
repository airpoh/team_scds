import pickle
import re
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class SpamModel:
    def __init__(self, model_dir="saved_models"):
        self.model_dir = model_dir
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.vectorizer = None
        self.model = None
        self._load_models()

    def _load_models(self):
        try:
            with open(os.path.join(self.model_dir, "vectorizer.pkl"), "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(os.path.join(self.model_dir, "rf_model.pkl"), "rb") as f:
                self.model = pickle.load(f)
            print("Spam model loaded successfully")
        except Exception as e:
            print(f"Error loading spam model: {e}")

    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
        text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation/numbers
        words = text.split()
        words = [self.stemmer.stem(w) for w in words if w not in self.stop_words]
        return " ".join(words)

    def predict(self, text: str):
        cleaned = self.clean_text(text)
        X_input = self.vectorizer.transform([cleaned])
        pred_label = self.model.predict(X_input)[0]
        pred_prob = self.model.predict_proba(X_input)[0][1]
        return pred_label, pred_prob
