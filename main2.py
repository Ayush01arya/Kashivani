#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VATIKA Tourism Chatbot for Varanasi
A Hindi-language tourism chatbot using open-source LLMs
"""

import json
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime

# For open-source LLM integration
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install: pip install transformers sentence-transformers torch")


@dataclass
class VATIKAEntry:
    """Data structure for VATIKA dataset entries"""
    domain: str
    context: str
    question: str
    answer: str
    entry_id: str


class VATIKADataProcessor:
    """Handles VATIKA dataset processing and preparation"""

    def __init__(self):
        self.domains = [
            "ganga_aarti", "cruise", "food_court", "public_toilet",
            "kund", "museum", "general", "ashram", "temple", "travel"
        ]
        self.domain_mappings = {
            "गंगा आरती": "ganga_aarti",
            "क्रूज़": "cruise",
            "फूड कोर्ट": "food_court",
            "सार्वजनिक शौचालय": "public_toilet",
            "कुंड": "kund",
            "संग्रहालय": "museum",
            "सामान्य": "general",
            "आश्रम": "ashram",
            "मंदिर": "temple",
            "यात्रा": "travel"
        }

    def load_vatika_dataset(self, file_path: str) -> List[VATIKAEntry]:
        """Load and parse VATIKA JSON dataset"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            entries = []
            for domain_data in data.get('domains', []):
                domain = domain_data['domain']

                for context_data in domain_data.get('contexts', []):
                    context = context_data['context']

                    for qa in context_data.get('qas', []):
                        entry = VATIKAEntry(
                            domain=domain,
                            context=context,
                            question=qa['question'],
                            answer=qa['answer'],
                            entry_id=qa['id']
                        )
                        entries.append(entry)

            logging.info(f"Loaded {len(entries)} entries from VATIKA dataset")
            return entries

        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            return []

    def preprocess_text(self, text: str) -> str:
        """Preprocess Hindi text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep Hindi characters
        text = re.sub(r'[^\u0900-\u097F\s\w\.\,\?\!\-\:\;]', '', text)
        return text

    def create_domain_embeddings(self, entries: List[VATIKAEntry]) -> Dict:
        """Create embeddings for domain-wise context retrieval"""
        domain_data = {}

        for domain in self.domains:
            domain_entries = [e for e in entries if e.domain == domain]
            if domain_entries:
                contexts = [self.preprocess_text(e.context) for e in domain_entries]
                questions = [self.preprocess_text(e.question) for e in domain_entries]

                domain_data[domain] = {
                    'entries': domain_entries,
                    'contexts': contexts,
                    'questions': questions
                }

        return domain_data


class VATIKAChatbot:
    """Main chatbot class for VATIKA tourism assistance"""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.data_processor = VATIKADataProcessor()
        self.entries = []
        self.domain_data = {}
        self.sentence_model = None
        self.vectorizer = None
        self.context_vectors = None

        # Common greetings and responses
        self.greetings = {
            "hi": "नमस्ते! मैं VATIKA हूँ, आपका वाराणसी टूरिज्म सहायक। मैं आपकी वाराणसी यात्रा में सहायता कर सकता हूँ।",
            "hello": "नमस्कार! वाराणसी के बारे में कोई भी सवाल पूछें।",
            "नमस्ते": "नमस्ते! मैं आपकी वाराणसी यात्रा में कैसे सहायता कर सकता हूँ?",
            "नमस्कार": "नमस्कार! वाराणसी के घाट, मंदिर, या अन्य स्थानों के बारे में पूछें।"
        }

        self.fallback_responses = [
            "मुझे खुशी होगी यदि आप अपना सवाल दूसरे तरीके से पूछें।",
            "क्या आप अपने सवाल को थोड़ा और स्पष्ट कर सकते हैं?",
            "वाराणसी के बारे में कुछ और पूछना चाहेंगे?",
            "मैं वाराणसी के घाट, मंदिर, भोजन, और यात्रा के बारे में जानकारी दे सकता हूँ।"
        ]

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def initialize_models(self):
        """Initialize the open-source models"""
        try:
            # Initialize sentence transformer for embeddings
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logging.info("Sentence transformer model loaded successfully")

            # Initialize TF-IDF vectorizer for backup retrieval
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                lowercase=True
            )

            return True
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
            return False

    def load_data(self, dataset_path: str):
        """Load and process VATIKA dataset"""
        self.entries = self.data_processor.load_vatika_dataset(dataset_path)
        if not self.entries:
            logging.error("No data loaded from dataset")
            return False

        self.domain_data = self.data_processor.create_domain_embeddings(self.entries)

        # Create embeddings for all contexts
        all_contexts = [entry.context for entry in self.entries]
        if self.sentence_model:
            self.context_vectors = self.sentence_model.encode(all_contexts)
        else:
            # Fallback to TF-IDF
            self.context_vectors = self.vectorizer.fit_transform(all_contexts)

        logging.info(f"Processed {len(self.entries)} entries across {len(self.domain_data)} domains")
        return True

    def classify_domain(self, question: str) -> str:
        """Classify question into appropriate domain"""
        question_lower = question.lower()

        # Domain keywords mapping
        domain_keywords = {
            "ganga_aarti": ["गंगा", "आरती", "aarti", "ganga", "ceremony"],
            "cruise": ["क्रूज़", "cruise", "boat", "नाव"],
            "food_court": ["खाना", "भोजन", "food", "restaurant", "रेस्टोरेंट"],
            "public_toilet": ["शौचालय", "toilet", "bathroom"],
            "kund": ["कुंड", "kund", "tank"],
            "museum": ["संग्रहालय", "museum", "म्यूजियम"],
            "ashram": ["आश्रम", "ashram"],
            "temple": ["मंदिर", "temple", "मन्दिर"],
            "travel": ["यात्रा", "travel", "जाना", "पहुंचना", "कैसे", "how to reach"],
            "general": ["सामान्य", "general", "about", "के बारे में"]
        }

        # Score each domain
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)

        return "general"

    def retrieve_relevant_context(self, question: str, top_k: int = 3) -> List[Tuple[VATIKAEntry, float]]:
        """Retrieve most relevant contexts for the question"""
        if not self.sentence_model or not self.context_vectors.any():
            return []

        # Encode the question
        question_vector = self.sentence_model.encode([question])

        # Calculate similarities
        similarities = cosine_similarity(question_vector, self.context_vectors)[0]

        # Get top-k most similar contexts
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Threshold for relevance
                results.append((self.entries[idx], similarities[idx]))

        return results

    def generate_response(self, question: str) -> str:
        """Generate response using retrieval-augmented generation"""
        # Check for greetings
        question_lower = question.lower().strip()
        for greeting, response in self.greetings.items():
            if greeting in question_lower:
                return response

        # Retrieve relevant contexts
        relevant_contexts = self.retrieve_relevant_context(question)

        if not relevant_contexts:
            return np.random.choice(self.fallback_responses)

        # Use the most relevant context
        best_match = relevant_contexts[0][0]

        # Check if we have a direct answer
        if best_match.question.lower() in question_lower or question_lower in best_match.question.lower():
            return best_match.answer

        # Generate contextual response
        context_info = best_match.context
        answer_template = best_match.answer

        # Create a comprehensive response
        response = f"{answer_template}\n\nअतिरिक्त जानकारी: {context_info[:200]}..."

        return response

    def chat_interface(self):
        """Interactive chat interface"""
        print("=" * 60)
        print("🕉️  VATIKA - वाराणसी टूरिज्म चैटबॉट  🕉️")
        print("=" * 60)
        print("नमस्ते! मैं आपका वाराणसी टूरिज्म सहायक हूँ।")
        print("वाराणसी के घाट, मंदिर, भोजन, यात्रा के बारे में पूछें।")
        print("बाहर निकलने के लिए 'exit' या 'quit' टाइप करें।")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n🙏 आप: ").strip()

                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("🙏 धन्यवाद! आपकी वाराणसी यात्रा मंगलमय हो!")
                    break

                if not user_input:
                    continue

                # Generate response
                response = self.generate_response(user_input)
                print(f"\n🤖 VATIKA: {response}")

            except KeyboardInterrupt:
                print("\n🙏 धन्यवाद! आपकी वाराणसी यात्रा मंगलमय हो!")
                break
            except Exception as e:
                print(f"❌ त्रुटि: {e}")
                print("कृपया दोबारा कोशिश करें।")


class VATIKAEvaluator:
    """Evaluation module for VATIKA chatbot"""

    def __init__(self, chatbot: VATIKAChatbot):
        self.chatbot = chatbot

    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score for response quality"""
        # Simplified BLEU calculation
        ref_tokens = reference.split()
        cand_tokens = candidate.split()

        if not cand_tokens:
            return 0.0

        # 1-gram precision
        ref_1grams = set(ref_tokens)
        cand_1grams = set(cand_tokens)

        precision = len(ref_1grams.intersection(cand_1grams)) / len(cand_1grams)
        return precision

    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L score"""
        ref_tokens = reference.split()
        cand_tokens = candidate.split()

        if not ref_tokens or not cand_tokens:
            return 0.0

        # Find LCS length
        lcs_length = self._lcs_length(ref_tokens, cand_tokens)

        if lcs_length == 0:
            return 0.0

        precision = lcs_length / len(cand_tokens)
        recall = lcs_length / len(ref_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def evaluate_on_test_data(self, test_entries: List[VATIKAEntry]) -> Dict[str, float]:
        """Evaluate chatbot on test dataset"""
        total_entries = len(test_entries)
        if total_entries == 0:
            return {"error": "No test data provided"}

        bleu_scores = []
        rouge_scores = []
        exact_matches = 0

        for entry in test_entries:
            # Generate response
            generated_response = self.chatbot.generate_response(entry.question)

            # Calculate metrics
            bleu = self.calculate_bleu_score(entry.answer, generated_response)
            rouge = self.calculate_rouge_l(entry.answer, generated_response)

            bleu_scores.append(bleu)
            rouge_scores.append(rouge)

            # Check for exact match
            if entry.answer.strip().lower() == generated_response.strip().lower():
                exact_matches += 1

        results = {
            "total_entries": total_entries,
            "avg_bleu_score": np.mean(bleu_scores),
            "avg_rouge_score": np.mean(rouge_scores),
            "exact_match_accuracy": exact_matches / total_entries,
            "bleu_std": np.std(bleu_scores),
            "rouge_std": np.std(rouge_scores)
        }

        return results


def main():
    """Main function to run the VATIKA chatbot"""
    # Initialize chatbot
    chatbot = VATIKAChatbot()

    # Initialize models
    if not chatbot.initialize_models():
        print("❌ Failed to initialize models. Please check your installation.")
        return

    # Create sample data structure based on provided sample
    sample_data = {
        "domains": [
            {
                "domain": "kund",
                "contexts": [
                    {
                        "context": "भागीरथ कुंड पं. दीन दयाल उपाध्याय रेलवे स्टेशन से 14.1 किलोमीटर दूर है। स्टेशन से कुंड तक पहुँचने के लिए टैक्सी, कैब, या बस सेवाओं का उपयोग किया जा सकता है। यह स्टेशन पूर्व में मुगलसराय के नाम से जाना जाता था और भारत के प्रमुख रेल जंक्शनों में से एक है।",
                        "qas": [
                            {
                                "id": "kund_636",
                                "question": "भागीरथ कुंड पं. दीन दयाल उपाध्याय रेलवे स्टेशन से कितना किलोमीटर दूर है?",
                                "answer": "भागीरथ कुंड पं. दीन दयाल उपाध्याय रेलवे स्टेशन से 14.1 किलोमीटर दूर है।"
                            },
                            {
                                "id": "kund_637",
                                "question": "भागीरथ कुंड पं. दीन दयाल उपाध्याय रेलवे स्टेशन से कैसे पहुँच सकते है।",
                                "answer": "भागीरथ कुंड तक पहुँचने के लिए पं. दीन दयाल उपाध्याय रेलवे स्टेशन से टैक्सी, कैब, या बस सेवाओं का उपयोग किया जा सकता है।"
                            }
                        ]
                    }
                ]
            }
        ]
    }

    # Save sample data for testing
    with open('train.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    # Load data
    if not chatbot.load_data('train.json'):
        print("❌ Failed to load dataset. Please check the file path.")
        return

    print("✅ VATIKA chatbot initialized successfully!")

    # Start chat interface
    chatbot.chat_interface()


if __name__ == "__main__":
    main()