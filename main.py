import json
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pickle
import os
import difflib


class ImprovedHindiQAChatbot:
    def __init__(self):
        # Enhanced TF-IDF with better parameters for Hindi
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=15000,
            min_df=2,
            max_df=0.85,
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b',  # Better for Hindi
            sublinear_tf=True,
            norm='l2'
        )

        self.qa_pairs = []
        self.qa_vectors = None
        self.question_answer_map = {}
        self.context_qa_map = defaultdict(list)
        self.domain_qa_map = defaultdict(list)

        # Hindi stop words (basic set)
        self.hindi_stop_words = {
            'का', 'के', 'की', 'को', 'से', 'में', 'पर', 'है', 'हैं', 'था', 'थे', 'थी',
            'होगा', 'होगी', 'होंगे', 'और', 'या', 'तो', 'भी', 'जो', 'वो', 'यह', 'वह',
            'इस', 'उस', 'कि', 'जब', 'तब', 'अब', 'फिर', 'यदि', 'तो', 'लेकिन',
            'परंतु', 'किंतु', 'अथवा', 'एवं', 'तथा'
        }

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for Hindi"""
        if not text:
            return ""

        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)

        # Normalize spaces
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove unwanted characters but keep Hindi, English, numbers, and basic punctuation
        text = re.sub(r'[^\u0900-\u097F\w\s\?\!\.\,\;\:]', ' ', text)

        # Remove extra punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\?\!]{2,}', '?', text)

        # Normalize common Hindi variations
        text = text.replace('ं', 'ं').replace('ँ', 'ँ')  # Normalize nasalization

        return text.strip()

    def remove_stop_words(self, text: str) -> str:
        """Remove Hindi stop words"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.hindi_stop_words]
        return ' '.join(filtered_words)

    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Remove stop words
        clean_text = self.remove_stop_words(text)

        # Split into words
        words = clean_text.split()

        # Filter out very short words (less than 2 characters)
        keywords = [word for word in words if len(word) >= 2]

        return keywords

    def load_training_data(self, json_files: List[str]):
        """Load and process training data from multiple JSON files"""
        all_qa_pairs = []

        for json_file in json_files:
            if not os.path.exists(json_file):
                print(f"Warning: File {json_file} not found, skipping...")
                continue

            print(f"Loading data from {json_file}...")

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for domain_data in data['domains']:
                domain = domain_data['domain']

                for context_data in domain_data['contexts']:
                    context = self.preprocess_text(context_data['context'])

                    for qa in context_data['qas']:
                        qa_item = {
                            'id': qa['id'],
                            'domain': domain,
                            'context': context,
                            'question': self.preprocess_text(qa['question']),
                            'answer': self.preprocess_text(qa['answer']),
                            'question_keywords': self.extract_keywords(qa['question']),
                            'answer_keywords': self.extract_keywords(qa['answer'])
                        }

                        all_qa_pairs.append(qa_item)

                        # Create mappings for faster retrieval
                        self.question_answer_map[qa_item['question']] = qa_item['answer']
                        self.context_qa_map[context].append(qa_item)
                        self.domain_qa_map[domain].append(qa_item)

        self.qa_pairs = all_qa_pairs
        print(f"Total QA pairs loaded: {len(self.qa_pairs)}")

    def create_enhanced_features(self, qa_item: Dict) -> str:
        """Create enhanced features combining question, context, and keywords"""
        # Combine question with important context information
        question = qa_item['question']
        context_snippet = qa_item['context'][:200]  # First 200 chars of context
        keywords = ' '.join(qa_item['question_keywords'])

        # Create weighted feature string
        enhanced_text = f"{question} {question} {keywords} {context_snippet}"
        return enhanced_text

    def train_model(self):
        """Train the enhanced retrieval model"""
        if not self.qa_pairs:
            raise ValueError("No training data loaded. Please load data first.")

        print("Creating enhanced features...")

        # Create enhanced feature vectors
        enhanced_texts = []
        for qa_item in self.qa_pairs:
            enhanced_text = self.create_enhanced_features(qa_item)
            enhanced_texts.append(enhanced_text)

        print("Training TF-IDF vectorizer...")
        self.qa_vectors = self.vectorizer.fit_transform(enhanced_texts)

        print("Model training completed!")

    def find_exact_matches(self, query: str) -> List[Tuple[Dict, float]]:
        """Find exact or near-exact matches"""
        query_lower = query.lower()
        exact_matches = []

        for qa_item in self.qa_pairs:
            question_lower = qa_item['question'].lower()

            # Check for exact match
            if query_lower == question_lower:
                exact_matches.append((qa_item, 1.0))
            # Check for very high similarity using difflib
            elif difflib.SequenceMatcher(None, query_lower, question_lower).ratio() > 0.9:
                similarity = difflib.SequenceMatcher(None, query_lower, question_lower).ratio()
                exact_matches.append((qa_item, similarity))

        return sorted(exact_matches, key=lambda x: x[1], reverse=True)

    def find_keyword_matches(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Find matches based on keyword overlap"""
        query_keywords = set(self.extract_keywords(query))

        if not query_keywords:
            return []

        keyword_matches = []

        for qa_item in self.qa_pairs:
            question_keywords = set(qa_item['question_keywords'])
            answer_keywords = set(qa_item['answer_keywords'])

            # Calculate keyword overlap
            question_overlap = len(query_keywords.intersection(question_keywords))
            answer_overlap = len(query_keywords.intersection(answer_keywords))

            # Combined score (weighted towards question keywords)
            total_keywords = len(query_keywords)
            if total_keywords > 0:
                keyword_score = (question_overlap * 0.7 + answer_overlap * 0.3) / total_keywords

                if keyword_score > 0.1:  # Minimum threshold
                    keyword_matches.append((qa_item, keyword_score))

        return sorted(keyword_matches, key=lambda x: x[1], reverse=True)[:top_k]

    def find_similar_qa(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Enhanced similarity search with multiple strategies"""
        query_processed = self.preprocess_text(query)

        # Strategy 1: Check for exact matches first
        exact_matches = self.find_exact_matches(query_processed)
        if exact_matches:
            return exact_matches[:top_k]

        # Strategy 2: TF-IDF similarity
        enhanced_query = f"{query_processed} {query_processed} {' '.join(self.extract_keywords(query_processed))}"
        query_vector = self.vectorizer.transform([enhanced_query])
        similarities = cosine_similarity(query_vector, self.qa_vectors).flatten()

        # Get top candidates
        top_indices = similarities.argsort()[-top_k * 2:][::-1]  # Get more candidates

        tfidf_results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                tfidf_results.append((self.qa_pairs[idx], similarities[idx]))

        # Strategy 3: Keyword-based matching
        keyword_results = self.find_keyword_matches(query_processed, top_k)

        # Combine and rank results
        combined_results = {}

        # Add TF-IDF results with weight
        for qa_item, score in tfidf_results:
            qa_id = qa_item['id']
            combined_results[qa_id] = {
                'qa_item': qa_item,
                'tfidf_score': score,
                'keyword_score': 0.0,
                'combined_score': score * 0.6
            }

        # Add keyword results with weight
        for qa_item, score in keyword_results:
            qa_id = qa_item['id']
            if qa_id in combined_results:
                combined_results[qa_id]['keyword_score'] = score
                combined_results[qa_id]['combined_score'] += score * 0.4
            else:
                combined_results[qa_id] = {
                    'qa_item': qa_item,
                    'tfidf_score': 0.0,
                    'keyword_score': score,
                    'combined_score': score * 0.4
                }

        # Sort by combined score
        final_results = []
        for qa_id, result_data in combined_results.items():
            final_results.append((result_data['qa_item'], result_data['combined_score']))

        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:top_k]

    def generate_answer(self, query: str) -> str:
        """Generate answer with improved confidence scoring"""
        if not query.strip():
            return "कृपया एक वैध प्रश्न पूछें।"

        similar_qa = self.find_similar_qa(query, top_k=3)

        if not similar_qa:
            return "मुझे खुशी होगी यदि आप अपना प्रश्न अधिक स्पष्ट रूप से पूछें। मैं आपकी बेहतर सहायता करने की कोशिश करूंगा।"

        best_match = similar_qa[0]
        best_qa, confidence = best_match

        # Dynamic confidence thresholds
        if confidence > 0.8:
            return best_qa['answer']
        elif confidence > 0.5:
            # High confidence - return answer with slight modification
            return best_qa['answer']
        elif confidence > 0.3:
            # Medium confidence - return answer with context
            return f"{best_qa['answer']}"
        else:
            # Low confidence - suggest rephrasing
            return "मैं आपके प्रश्न को पूरी तरह से समझ नहीं पाया। कृपया अपना प्रश्न थोड़ा अलग तरीके से पूछें।"

    def batch_predict(self, questions: List[str]) -> List[str]:
        """Predict answers for multiple questions efficiently"""
        answers = []
        for question in questions:
            answer = self.generate_answer(question)
            answers.append(answer)
        return answers

    def save_model(self, model_path: str):
        """Save the trained model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'qa_pairs': self.qa_pairs,
            'qa_vectors': self.qa_vectors,
            'question_answer_map': self.question_answer_map,
            'context_qa_map': dict(self.context_qa_map),
            'domain_qa_map': dict(self.domain_qa_map),
            'hindi_stop_words': self.hindi_stop_words
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load a pre-trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data['vectorizer']
        self.qa_pairs = model_data['qa_pairs']
        self.qa_vectors = model_data['qa_vectors']
        self.question_answer_map = model_data['question_answer_map']
        self.context_qa_map = defaultdict(list, model_data['context_qa_map'])
        self.domain_qa_map = defaultdict(list, model_data['domain_qa_map'])
        self.hindi_stop_words = model_data.get('hindi_stop_words', self.hindi_stop_words)

        print(f"Model loaded from {model_path}")


class ModelEvaluator:
    """Enhanced evaluation metrics"""

    @staticmethod
    def calculate_exact_match(predicted: str, actual: str) -> float:
        """Calculate exact match score"""
        return 1.0 if predicted.strip().lower() == actual.strip().lower() else 0.0

    @staticmethod
    def calculate_word_overlap_f1(predicted: str, actual: str) -> float:
        """Calculate F1 score based on word overlap"""
        pred_words = set(predicted.lower().split())
        actual_words = set(actual.lower().split())

        if len(pred_words) == 0 and len(actual_words) == 0:
            return 1.0
        if len(pred_words) == 0 or len(actual_words) == 0:
            return 0.0

        common = pred_words.intersection(actual_words)

        if len(common) == 0:
            return 0.0

        precision = len(common) / len(pred_words)
        recall = len(common) / len(actual_words)

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def evaluate_model(chatbot, validation_file: str, sample_size: int = None):
        """Comprehensive model evaluation"""
        with open(validation_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)

        questions = []
        actual_answers = []

        for domain_data in val_data['domains']:
            for context_data in domain_data['contexts']:
                for qa in context_data['qas']:
                    questions.append(qa['question'])
                    actual_answers.append(qa['answer'])

        # Sample data if specified
        if sample_size and sample_size < len(questions):
            indices = np.random.choice(len(questions), sample_size, replace=False)
            questions = [questions[i] for i in indices]
            actual_answers = [actual_answers[i] for i in indices]

        print(f"Evaluating on {len(questions)} questions...")

        # Generate predictions
        predicted_answers = chatbot.batch_predict(questions)

        # Calculate metrics
        exact_matches = []
        f1_scores = []

        for pred, actual in zip(predicted_answers, actual_answers):
            exact_match = ModelEvaluator.calculate_exact_match(pred, actual)
            f1_score = ModelEvaluator.calculate_word_overlap_f1(pred, actual)

            exact_matches.append(exact_match)
            f1_scores.append(f1_score)

        avg_exact_match = np.mean(exact_matches)
        avg_f1 = np.mean(f1_scores)

        print(f"\n=== Evaluation Results ===")
        print(f"Exact Match Accuracy: {avg_exact_match:.4f}")
        print(f"Average F1 Score: {avg_f1:.4f}")
        print(f"Total Questions: {len(questions)}")

        return {
            'exact_match': avg_exact_match,
            'f1_score': avg_f1,
            'total_questions': len(questions)
        }


def train_and_evaluate():
    """Complete training and evaluation pipeline"""

    # Initialize chatbot
    chatbot = ImprovedHindiQAChatbot()

    # Load training data (add all available files)
    training_files = ['train.json']  # Add 'validation.json' if you want to use it for training too

    # Check which files exist
    available_files = [f for f in training_files if os.path.exists(f)]

    if not available_files:
        print("Error: No training data files found!")
        print("Please ensure 'train.json' exists in the current directory.")
        return None

    # Load and train
    chatbot.load_training_data(available_files)
    chatbot.train_model()

    # Evaluate if validation file exists
    if os.path.exists('validation.json'):
        print("\n=== Evaluating Model ===")
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(chatbot, 'validation.json', sample_size=100)

    # Save the model
    chatbot.save_model('optimized_hindi_qa_model.pkl')

    # Interactive testing
    print("\n=== Interactive Testing ===")
    print("Enter questions to test the model (type 'quit' to exit):")

    while True:
        question = input("\nप्रश्न: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if question:
            answer = chatbot.generate_answer(question)
            print(f"उत्तर: {answer}")

    return chatbot


def predict_test_data(model_path: str, test_file: str, output_file: str):
    """Generate predictions for test data"""

    # Load trained model
    chatbot = ImprovedHindiQAChatbot()
    chatbot.load_model(model_path)

    # Load test data
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # Generate predictions
    for domain_data in test_data['domains']:
        for context_data in domain_data['contexts']:
            for qa in context_data['qas']:
                question = qa['question']
                predicted_answer = chatbot.generate_answer(question)
                qa['answer'] = predicted_answer

    # Save predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    # Train and evaluate the model
    trained_model = train_and_evaluate()

    # When you get test data, use this:
    # predict_test_data('optimized_hindi_qa_model.pkl', 'test_data.json', 'predictions.json')