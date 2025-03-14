import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk
from transformers import MarianMTModel, MarianTokenizer

class DataProcessor:
    def __init__(self, tokenizer_name: str = "EleutherAI/gpt-j-6B"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Download required NLTK data
            nltk.download('punkt')
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')
            
            # Initialize translation models for backtranslation
            self.en_fr_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
            self.fr_en_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
            self.en_fr_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
            self.fr_en_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DataProcessor: {str(e)}")
        
    def load_and_preprocess(self, 
                           dataset_path: str,
                           text_column: str = "text",
                           max_length: int = 512,
                           train_size: float = 0.8) -> Dict[str, Dataset]:
        """Load and preprocess the dataset"""
        try:
            # Validate inputs
            if not dataset_path:
                raise ValueError("dataset_path cannot be empty")
            if not 0 < train_size < 1:
                raise ValueError("train_size must be between 0 and 1")
            
            # Load raw data
            if os.path.isfile(dataset_path):
                # Load from local file
                if dataset_path.endswith('.csv'):
                    df = pd.read_csv(dataset_path)
                    dataset = Dataset.from_pandas(df)
                elif dataset_path.endswith('.json'):
                    dataset = load_dataset('json', data_files=dataset_path)
                else:
                    raise ValueError(f"Unsupported file format: {dataset_path}")
            else:
                # Try loading from Hugging Face datasets
                dataset = load_dataset(dataset_path)
            
            if text_column not in dataset["train"].column_names:
                raise ValueError(f"Text column '{text_column}' not found in dataset")
            
            # Basic cleaning
            cleaned_texts = self._clean_texts(dataset["train"][text_column])
            
            # Create train/val split
            train_texts, val_texts = train_test_split(
                cleaned_texts,
                train_size=train_size,
                random_state=42
            )
            
            # Tokenize datasets
            train_dataset = self._tokenize_texts(train_texts, max_length)
            val_dataset = self._tokenize_texts(val_texts, max_length)
            
            return {
                "train": train_dataset,
                "validation": val_dataset
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load and preprocess dataset: {str(e)}")
    
    def _clean_texts(self, texts: List[str]) -> List[str]:
        """Comprehensive text cleaning"""
        cleaned = []
        for text in texts:
            try:
                # Remove extra whitespace
                text = " ".join(text.split())
                # Remove special characters while keeping punctuation
                text = "".join(char for char in text if char.isprintable())
                # Basic normalization
                text = text.strip().lower()
                # Remove empty or very short texts
                if len(text) > 10:  # Minimum length threshold
                    cleaned.append(text)
            except Exception as e:
                print(f"Warning: Failed to clean text: {str(e)}")
                continue
        return cleaned
    
    def _tokenize_texts(self, texts: List[str], max_length: int) -> Dataset:
        """Tokenize texts and create dataset with error handling"""
        try:
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            
            return Dataset.from_dict({
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"]
            })
        except Exception as e:
            raise RuntimeError(f"Failed to tokenize texts: {str(e)}")
    
    def augment_data(self, 
                    texts: List[str],
                    techniques: Optional[List[str]] = None) -> List[str]:
        """Apply data augmentation techniques"""
        if techniques is None:
            techniques = ["backtranslation", "synonym_replacement"]
            
        augmented_texts = []
        for text in texts:
            augmented_texts.append(text)  # Original text
            
            try:
                if "synonym_replacement" in techniques:
                    augmented = self._synonym_replacement(text)
                    if augmented and augmented != text:
                        augmented_texts.append(augmented)
                    
                if "backtranslation" in techniques:
                    augmented = self._backtranslation(text)
                    if augmented and augmented != text:
                        augmented_texts.append(augmented)
            except Exception as e:
                print(f"Warning: Failed to augment text: {str(e)}")
                continue
                
        return augmented_texts
    
    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms using WordNet"""
        try:
            words = word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            
            # Only replace content words (nouns, verbs, adjectives)
            replaceable_pos = {'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ'}
            
            for i, (word, pos) in enumerate(pos_tags):
                if pos[:2] in replaceable_pos:
                    synsets = wordnet.synsets(word)
                    if synsets:
                        # Get all lemmas from all synsets
                        lemmas = [lemma.name() for synset in synsets for lemma in synset.lemmas()]
                        # Remove duplicates and original word
                        lemmas = list(set(lemmas))
                        if word in lemmas:
                            lemmas.remove(word)
                        if lemmas:
                            words[i] = np.random.choice(lemmas)
            
            return " ".join(words)
        except Exception as e:
            print(f"Warning: Synonym replacement failed: {str(e)}")
            return text
    
    def _backtranslation(self, text: str) -> str:
        """Implement backtranslation using MarianMT"""
        try:
            # English to French
            fr_tokens = self.en_fr_tokenizer(text, return_tensors="pt", padding=True)
            fr_translation = self.en_fr_model.generate(**fr_tokens)
            fr_text = self.en_fr_tokenizer.decode(fr_translation[0], skip_special_tokens=True)
            
            # French back to English
            en_tokens = self.fr_en_tokenizer(fr_text, return_tensors="pt", padding=True)
            en_translation = self.fr_en_model.generate(**en_tokens)
            en_text = self.fr_en_tokenizer.decode(en_translation[0], skip_special_tokens=True)
            
            return en_text
        except Exception as e:
            print(f"Warning: Backtranslation failed: {str(e)}")
            return text
