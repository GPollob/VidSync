"""
AikoInfinity 2.0 Text Preprocessor - AI Engine
This script contains functions for preprocessing text data before feeding it into AI models.
"""

import re
import string
import nltk
import torch
from transformers import GPT2Tokenizer, BertTokenizer
from typing import List, Tuple
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger("AikoTextPreprocessor")

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """
    Class for preprocessing text data, including tokenization, cleaning, and other necessary steps.
    """

    def __init__(self, model_type: str = 'gpt2'):
        """
        Initializes the text preprocessor based on the selected model type.
        
        :param model_type: The type of model for which to preprocess text ('gpt2', 'bert', etc.)
        """
        self.model_type = model_type
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        if model_type == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        elif model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            raise ValueError("Unsupported model type. Please choose either 'gpt2' or 'bert'.")

        logger.info(f"TextPreprocessor initialized for {model_type}.")

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text by removing special characters, extra spaces, and converting to lowercase.
        
        :param text: The raw input text to be cleaned.
        :return: Cleaned text.
        """
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
        logger.debug(f"Cleaned text: {text}")
        return text

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenizes the input text using word tokenization.

        :param text: The cleaned text to be tokenized.
        :return: List of tokens.
        """
        tokens = word_tokenize(text)
        logger.debug(f"Tokenized text: {tokens}")
        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Removes stopwords (common words) from a list of tokens.
        
        :param tokens: The list of tokens from which to remove stopwords.
        :return: List of tokens without stopwords.
        """
        tokens_without_stopwords = [word for word in tokens if word not in self.stop_words]
        logger.debug(f"Tokens without stopwords: {tokens_without_stopwords}")
        return tokens_without_stopwords

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatizes the tokens to get their root forms (e.g., "running" -> "run").
        
        :param tokens: The list of tokens to be lemmatized.
        :return: Lemmatized list of tokens.
        """
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        logger.debug(f"Lemmatized tokens: {lemmatized_tokens}")
        return lemmatized_tokens

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocesses the input text by cleaning, tokenizing, removing stopwords, and lemmatizing.
        
        :param text: The raw input text.
        :return: List of lemmatized tokens.
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        tokens_without_stopwords = self.remove_stopwords(tokens)
        lemmatized_tokens = self.lemmatize_tokens(tokens_without_stopwords)
        logger.info(f"Preprocessed text: {lemmatized_tokens}")
        return lemmatized_tokens

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encodes the input text into token IDs using the appropriate tokenizer.
        
        :param text: The cleaned and preprocessed text.
        :return: A tensor containing token IDs.
        """
        tokens = self.preprocess_text(text)
        encoded_input = self.tokenizer(tokens, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded_input['input_ids']
        logger.debug(f"Encoded text (input IDs): {input_ids}")
        return input_ids

    def decode_text(self, input_ids: torch.Tensor) -> str:
        """
        Decodes the token IDs back into human-readable text.
        
        :param input_ids: The token IDs to decode.
        :return: The decoded text.
        """
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        logger.debug(f"Decoded text: {decoded_text}")
        return decoded_text


def test_preprocessing():
    """
    Function to test the text preprocessor on a sample input text.
    """
    text = """
    AikoInfinity 2.0 is an innovative AI-driven platform built for the future of immersive technologies.
    """
    preprocessor = TextPreprocessor(model_type='gpt2')
    preprocessed_text = preprocessor.preprocess_text(text)
    print("Preprocessed Text:", preprocessed_text)

    encoded_text = preprocessor.encode_text(text)
    print("Encoded Text (Tensor):", encoded_text)

    decoded_text = preprocessor.decode_text(encoded_text)
    print("Decoded Text:", decoded_text)


if __name__ == "__main__":
    test_preprocessing()
