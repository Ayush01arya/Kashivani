import streamlit as st
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime
import pickle
import os
import time
from dataclasses import dataclass, asdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import hashlib
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="VATIKA - Enhanced ML Pipeline",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #FF6B35;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #0066cc;
    }
    .bot-message {
        background-color: #fff5f0;
        border-left-color: #FF6B35;
    }
    .domain-tag {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        background-color: #FF6B35;
        color: white;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .metrics-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #28a745;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #dc3545;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class VATIKAData:
    """Enhanced data structure for VATIKA entries"""
    id: str
    domain: str
    context: str
    question: str
    answer: str
    keywords: List[str] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = self.extract_keywords()

    def extract_keywords(self) -> List[str]:
        """Extract keywords from question and answer"""
        text = f"{self.question} {self.answer}".lower()
        # Simple keyword extraction - can be improved with NLP libraries
        words = re.findall(r'\b\w+\b', text)
        return [word for word in words if len(word) > 2]

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'domain': self.domain,
            'context': self.context,
            'question': self.question,
            'answer': self.answer,
            'keywords': self.keywords
        }

    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(
            id=data['id'],
            domain=data['domain'],
            context=data['context'],
            question=data['question'],
            answer=data['answer'],
            keywords=data.get('keywords', [])
        )


class EnhancedVATIKAModel:
    """Enhanced VATIKA ML Model with better validation and features"""

    def __init__(self):
        # Multiple vectorizers for better feature extraction
        self.question_vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),
            stop_words=None,
            lowercase=True,
            sublinear_tf=True,
            min_df=2,
            max_df=0.8
        )

        self.context_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )

        self.combined_vectorizer = TfidfVectorizer(
            max_features=4000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )

        # Data storage
        self.train_data: List[VATIKAData] = []
        self.val_data: List[VATIKAData] = []
        self.test_data: List[VATIKAData] = []

        # Trained components
        self.question_vectors_train = None
        self.context_vectors_train = None
        self.combined_vectors_train = None
        self.domain_classifier = None
        self.trained = False
        self.validation_metrics = {}

        # Model paths
        self.model_dir = "models"
        self.ensure_model_dir()

    def ensure_model_dir(self):
        """Ensure model directory exists"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
        except Exception as e:
            st.error(f"Cannot create models directory: {str(e)}")

    def load_and_split_data(self, json_file_path: str, train_split: float = 0.7, val_split: float = 0.15) -> Tuple[
        bool, str]:
        """Load data and split into train/validation/test sets"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Clear existing data
            all_data = []
            total_entries = 0

            # Process JSON structure (same as before)
            if not isinstance(data, dict) or 'domains' not in data:
                return False, "Invalid JSON structure"

            for domain_idx, domain_data in enumerate(data['domains']):
                if not isinstance(domain_data, dict):
                    continue

                domain_name = domain_data.get('domain', f'domain_{domain_idx}')

                if 'contexts' not in domain_data:
                    continue

                for context_idx, context_data in enumerate(domain_data['contexts']):
                    if not isinstance(context_data, dict):
                        continue

                    context = context_data.get('context', '')
                    if 'qas' not in context_data:
                        continue

                    for qa_idx, qa in enumerate(context_data['qas']):
                        if not isinstance(qa, dict):
                            continue

                        question = qa.get('question', '').strip()
                        answer = qa.get('answer', '').strip()

                        if not question or not answer:
                            continue

                        entry = VATIKAData(
                            id=qa.get('id', f"{domain_name}_{context_idx}_{qa_idx}"),
                            domain=domain_name,
                            context=context,
                            question=question,
                            answer=answer
                        )
                        all_data.append(entry)
                        total_entries += 1

            if total_entries < 10:
                return False, f"Insufficient data: need at least 10 entries, got {total_entries}"

            # Stratified split by domain
            domains = [entry.domain for entry in all_data]
            unique_domains = list(set(domains))

            if len(unique_domains) == 1:
                # Simple random split if only one domain
                train_data, temp_data = train_test_split(all_data, train_size=train_split, random_state=42)
                val_size = val_split / (1 - train_split)
                val_data, test_data = train_test_split(temp_data, train_size=val_size, random_state=42)
            else:
                # Stratified split
                train_data, temp_data = train_test_split(
                    all_data, train_size=train_split, stratify=domains, random_state=42
                )
                temp_domains = [entry.domain for entry in temp_data]
                val_size = val_split / (1 - train_split)
                val_data, test_data = train_test_split(
                    temp_data, train_size=val_size, stratify=temp_domains, random_state=42
                )

            self.train_data = train_data
            self.val_data = val_data
            self.test_data = test_data

            return True, f"Data loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test"

        except Exception as e:
            return False, f"Error loading data: {str(e)}"

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Keep Hindi, English, numbers, and basic punctuation
        text = re.sub(r'[^\u0900-\u097F\u0020-\u007E]', ' ', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text.strip())

        return text

    def extract_features(self, entries: List[VATIKAData]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract multiple types of features"""
        questions = [self.preprocess_text(entry.question) for entry in entries]
        contexts = [self.preprocess_text(entry.context) for entry in entries]
        combined = [f"{q} {c}" for q, c in zip(questions, contexts)]

        return questions, contexts, combined

    def train_model(self, progress_callback=None) -> Tuple[bool, str]:
        """Enhanced training with proper validation"""
        if not self.train_data:
            return False, "No training data loaded"

        try:
            start_time = time.time()
            total_steps = 10
            current_step = 0

            def update_progress(step_name: str):
                nonlocal current_step
                current_step += 1
                if progress_callback:
                    progress_callback(current_step, total_steps, step_name)

            # Step 1: Validate data
            update_progress("Validating data split...")
            if len(self.train_data) < 5:
                return False, f"Insufficient training data: {len(self.train_data)}"

            # Step 2: Extract features
            update_progress("Extracting training features...")
            train_questions, train_contexts, train_combined = self.extract_features(self.train_data)

            # Step 3: Train vectorizers
            update_progress("Training question vectorizer...")
            self.question_vectors_train = self.question_vectorizer.fit_transform(train_questions)

            update_progress("Training context vectorizer...")
            self.context_vectors_train = self.context_vectorizer.fit_transform(train_contexts)

            update_progress("Training combined vectorizer...")
            self.combined_vectors_train = self.combined_vectorizer.fit_transform(train_combined)

            # Step 4: Train domain classifier
            update_progress("Training domain classifier...")
            if len(set(entry.domain for entry in self.train_data)) > 1:
                # Combine features for domain classification
                domain_features = np.hstack([
                    self.question_vectors_train.toarray(),
                    self.context_vectors_train.toarray()
                ])
                domain_labels = [entry.domain for entry in self.train_data]

                self.domain_classifier = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10
                )
                self.domain_classifier.fit(domain_features, domain_labels)

            # Step 5: Validate on validation set
            update_progress("Validating model...")
            val_metrics = self.validate_model()
            self.validation_metrics = val_metrics

            # Step 6: Save vectorizers
            update_progress("Saving vectorizers...")
            self.save_vectorizers()

            # Step 7: Save training data
            update_progress("Saving training data...")
            self.save_training_data()

            # Step 8: Save vectors
            update_progress("Saving vectors...")
            self.save_vectors()

            # Step 9: Save domain classifier
            update_progress("Saving domain classifier...")
            if self.domain_classifier:
                joblib.dump(self.domain_classifier, os.path.join(self.model_dir, "domain_classifier.pkl"))

            # Step 10: Finalize
            update_progress("Finalizing model...")
            self.trained = True

            end_time = time.time()
            training_time = end_time - start_time

            # Prepare result message
            result_msg = f"""
            Model trained successfully!
            • Training time: {training_time:.2f}s
            • Training entries: {len(self.train_data)}
            • Validation entries: {len(self.val_data)}
            • Test entries: {len(self.test_data)}
            • Validation accuracy: {val_metrics.get('accuracy', 0):.3f}
            • Validation F1: {val_metrics.get('f1', 0):.3f}
            """

            return True, result_msg

        except Exception as e:
            self.trained = False
            return False, f"Training failed: {str(e)}"

    def validate_model(self) -> Dict:
        """Validate model on validation set"""
        if not self.val_data:
            return {"accuracy": 0, "f1": 0, "precision": 0, "recall": 0}

        try:
            correct_predictions = 0
            total_predictions = len(self.val_data)

            # For each validation example
            for val_entry in self.val_data:
                predictions = self.predict_single(val_entry.question, exclude_domains=None)

                if predictions:
                    best_pred, confidence = predictions[0]
                    # Check if prediction is from same domain and similar answer
                    if best_pred.domain == val_entry.domain:
                        # Simple similarity check
                        if self.calculate_answer_similarity(best_pred.answer, val_entry.answer) > 0.3:
                            correct_predictions += 1

            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

            return {
                "accuracy": accuracy,
                "f1": accuracy,  # Simplified for now
                "precision": accuracy,
                "recall": accuracy,
                "total_predictions": total_predictions,
                "correct_predictions": correct_predictions
            }

        except Exception as e:
            st.error(f"Validation error: {str(e)}")
            return {"accuracy": 0, "f1": 0, "precision": 0, "recall": 0}

    def calculate_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate similarity between two answers"""
        try:
            # Simple word overlap similarity
            words1 = set(answer1.lower().split())
            words2 = set(answer2.lower().split())

            if not words1 or not words2:
                return 0.0

            intersection = words1.intersection(words2)
            union = words1.union(words2)

            return len(intersection) / len(union) if union else 0.0

        except:
            return 0.0

    def predict_single(self, query: str, exclude_domains: List[str] = None) -> List[Tuple[VATIKAData, float]]:
        """Predict for a single query with improved scoring"""
        if not self.trained:
            return []

        try:
            processed_query = self.preprocess_text(query)
            if not processed_query.strip():
                return []

            # Vectorize query
            query_q_vector = self.question_vectorizer.transform([processed_query])
            query_c_vector = self.context_vectorizer.transform([processed_query])
            query_combined_vector = self.combined_vectorizer.transform([processed_query])

            # Calculate multiple similarities
            q_similarities = cosine_similarity(query_q_vector, self.question_vectors_train).flatten()
            c_similarities = cosine_similarity(query_c_vector, self.context_vectors_train).flatten()
            combined_similarities = cosine_similarity(query_combined_vector, self.combined_vectors_train).flatten()

            # Combine similarities with weights
            final_similarities = (
                    0.5 * q_similarities +
                    0.2 * c_similarities +
                    0.3 * combined_similarities
            )

            # Get top candidates
            top_indices = final_similarities.argsort()[-10:][::-1]

            results = []
            for idx in top_indices:
                if idx < len(self.train_data):
                    entry = self.train_data[idx]
                    score = final_similarities[idx]

                    # Skip if domain excluded
                    if exclude_domains and entry.domain in exclude_domains:
                        continue

                    # Only include if score is reasonable
                    if score > 0.05:  # Threshold for relevance
                        results.append((entry, score))

            # Limit results
            return results[:5]

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return []

    def predict(self, query: str, top_k: int = 5) -> List[Tuple[VATIKAData, float]]:
        """Main prediction method"""
        return self.predict_single(query)[:top_k]

    def save_vectorizers(self):
        """Save vectorizers"""
        joblib.dump(self.question_vectorizer, os.path.join(self.model_dir, "question_vectorizer.pkl"))
        joblib.dump(self.context_vectorizer, os.path.join(self.model_dir, "context_vectorizer.pkl"))
        joblib.dump(self.combined_vectorizer, os.path.join(self.model_dir, "combined_vectorizer.pkl"))

    def save_training_data(self):
        """Save training data as JSON"""
        data = {
            'train': [entry.to_dict() for entry in self.train_data],
            'val': [entry.to_dict() for entry in self.val_data],
            'test': [entry.to_dict() for entry in self.test_data]
        }

        with open(os.path.join(self.model_dir, "training_data.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_vectors(self):
        """Save pre-computed vectors"""
        vectors_data = {
            'question_vectors': self.question_vectors_train.toarray(),
            'context_vectors': self.context_vectors_train.toarray(),
            'combined_vectors': self.combined_vectors_train.toarray()
        }

        with open(os.path.join(self.model_dir, "vectors.pkl"), 'wb') as f:
            pickle.dump(vectors_data, f, protocol=4)

    def load_trained_model(self) -> Tuple[bool, str]:
        """Load pre-trained model"""
        try:
            # Load vectorizers
            self.question_vectorizer = joblib.load(os.path.join(self.model_dir, "question_vectorizer.pkl"))
            self.context_vectorizer = joblib.load(os.path.join(self.model_dir, "context_vectorizer.pkl"))
            self.combined_vectorizer = joblib.load(os.path.join(self.model_dir, "combined_vectorizer.pkl"))

            # Load training data
            with open(os.path.join(self.model_dir, "training_data.json"), 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.train_data = [VATIKAData.from_dict(item) for item in data['train']]
            self.val_data = [VATIKAData.from_dict(item) for item in data['val']]
            self.test_data = [VATIKAData.from_dict(item) for item in data['test']]

            # Load vectors
            with open(os.path.join(self.model_dir, "vectors.pkl"), 'rb') as f:
                vectors_data = pickle.load(f)

            self.question_vectors_train = vectors_data['question_vectors']
            self.context_vectors_train = vectors_data['context_vectors']
            self.combined_vectors_train = vectors_data['combined_vectors']

            # Load domain classifier if exists
            domain_classifier_path = os.path.join(self.model_dir, "domain_classifier.pkl")
            if os.path.exists(domain_classifier_path):
                self.domain_classifier = joblib.load(domain_classifier_path)

            self.trained = True
            return True, f"Model loaded: {len(self.train_data)} train, {len(self.val_data)} val, {len(self.test_data)} test"

        except Exception as e:
            return False, f"Error loading model: {str(e)}"

    def test_model(self) -> Dict:
        """Test model on test set"""
        if not self.test_data or not self.trained:
            return {}

        try:
            correct_predictions = 0
            total_predictions = len(self.test_data)

            for test_entry in self.test_data:
                predictions = self.predict_single(test_entry.question)

                if predictions:
                    best_pred, confidence = predictions[0]
                    if best_pred.domain == test_entry.domain:
                        if self.calculate_answer_similarity(best_pred.answer, test_entry.answer) > 0.3:
                            correct_predictions += 1

            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

            return {
                "test_accuracy": accuracy,
                "test_total": total_predictions,
                "test_correct": correct_predictions
            }

        except Exception as e:
            st.error(f"Testing error: {str(e)}")
            return {}

    def get_domain_statistics(self) -> Dict:
        """Get domain statistics"""
        all_data = self.train_data + self.val_data + self.test_data
        domain_stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

        for entry in self.train_data:
            domain_stats[entry.domain]["train"] += 1
        for entry in self.val_data:
            domain_stats[entry.domain]["val"] += 1
        for entry in self.test_data:
            domain_stats[entry.domain]["test"] += 1

        return dict(domain_stats)


class EnhancedVATIKAChatbot:
    """Enhanced chatbot with better response handling"""

    def __init__(self, model: EnhancedVATIKAModel):
        self.model = model
        self.conversation_history = []

    def chat(self, query: str) -> Tuple[str, str, float]:
        """Enhanced chat with better response selection"""
        if not self.model.trained:
            return "मॉडल अभी तक प्रशिक्षित नहीं है। कृपया पहले मॉडल को प्रशिक्षित करें।", "error", 0.0

        if not query.strip():
            return "कृपया एक वैध प्रश्न पूछें।", "error", 0.0

        predictions = self.model.predict(query, top_k=3)

        if not predictions:
            return "क्षमा करें, मुझे इस प्रश्न का उत्तर नहीं मिला। कृपया अन्य प्रश्न पूछें।", "unknown", 0.0

        best_prediction, confidence = predictions[0]

        # Enhanced response based on confidence
        if confidence < 0.1:
            response = f"मुझे पूरा विश्वास नहीं है, लेकिन संभावित उत्तर: {best_prediction.answer}"
        else:
            response = best_prediction.answer

        # Add to conversation history
        self.conversation_history.append({
            "query": query,
            "answer": response,
            "domain": best_prediction.domain,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        return response, best_prediction.domain, confidence


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🕉️ VATIKA - Enhanced ML Pipeline</h1>
        <p>Advanced Varanasi Tourism Question Answering System</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize model
    if 'enhanced_model' not in st.session_state:
        st.session_state.enhanced_model = EnhancedVATIKAModel()
        st.session_state.enhanced_chatbot = EnhancedVATIKAChatbot(st.session_state.enhanced_model)

    model = st.session_state.enhanced_model
    chatbot = st.session_state.enhanced_chatbot

    # Sidebar
    with st.sidebar:
        st.header("🛠️ Enhanced Model Control")

        # Model Status
        if model.trained:
            st.success("🟢 Model Status: TRAINED & READY")
            if model.validation_metrics:
                st.metric("Validation Accuracy", f"{model.validation_metrics.get('accuracy', 0):.3f}")
        else:
            st.error("🔴 Model Status: NOT TRAINED")

        # Data Statistics
        st.subheader("📊 Data Statistics")
        st.write(f"**Training:** {len(model.train_data)} entries")
        st.write(f"**Validation:** {len(model.val_data)} entries")
        st.write(f"**Test:** {len(model.test_data)} entries")

        if model.train_data:
            domain_stats = model.get_domain_statistics()
            st.write("**Domain Distribution:**")
            for domain, stats in domain_stats.items():
                st.write(f"  • {domain}: {stats['train']}|{stats['val']}|{stats['test']}")

        st.markdown("---")

        # Create enhanced sample data
        if st.button("📝 Create Enhanced Sample Data"):
            sample_data = {
                "domains": [
                    {
                        "domain": "temples",
                        "contexts": [
                            {
                                "context": "वाराणसी में कई प्रसिद्ध मंदिर हैं जो हिंदू धर्म के लिए महत्वपूर्ण हैं।",
                                "qas": [
                                    {
                                        "id": "1",
                                        "question": "काशी विश्वनाथ मंदिर कहाँ है?",
                                        "answer": "काशी विश्वनाथ मंदिर वाराणसी, उत्तर प्रदेश में स्थित है।"
                                    },
                                    {
                                        "id": "2",
                                        "question": "वाराणसी के प्रमुख मंदिर कौन से हैं?",
                                        "answer": "वाराणसी के प्रमुख मंदिर काशी विश्वनाथ, संकट मोचन हनुमान मंदिर, और दुर्गा मंदिर हैं।"
                                    },
                                    {
                                        "id": "3",
                                        "question": "काशी विश्वनाथ मंदिर का इतिहास क्या है?",
                                        "answer": "काशी विश्वनाथ मंदिर का इतिहास हजारों साल पुराना है और यह भगवान शिव का प्रमुख मंदिर है।"
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "domain": "food",
                        "contexts": [
                            {
                                "context": "वाराणसी अपने स्वादिष्ट स्ट्रीट फूड के लिए प्रसिद्ध है।",
                                "qas": [
                                    {
                                        "id": "4",
                                        "question": "वाराणसी का प्रसिद्ध भोजन क्या है?",
                                        "answer": "वाराणसी का प्रसिद्ध भोजन कचौड़ी-सब्जी, चाट, लस्सी और बनारसी पान है।"
                                    },
                                    {
                                        "id": "5",
                                        "question": "बनारसी पान क्यों प्रसिद्ध है?",
                                        "answer": "बनारसी पान अपने विशेष स्वाद और सुगंध के लिए प्रसिद्ध है।"
                                    },
                                    {
                                        "id": "6",
                                        "question": "वाराणसी में कहाँ खाना खाएं?",
                                        "answer": "वाराणसी में गोदौलिया मार्केट, विश्वनाथ गली और ठठेरी बाजार में अच्छा खाना मिलता है।"
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "domain": "ghats",
                        "contexts": [
                            {
                                "context": "वाराणसी में गंगा नदी के किनारे कई घाट हैं जो धार्मिक महत्व रखते हैं।",
                                "qas": [
                                    {
                                        "id": "7",
                                        "question": "दशाश्वमेध घाट क्यों प्रसिद्ध है?",
                                        "answer": "दशाश्वमेध घाट गंगा आरती के लिए प्रसिद्ध है और यह वाराणसी का मुख्य घाट है।"
                                    },
                                    {
                                        "id": "8",
                                        "question": "वाराणसी में कितने घाट हैं?",
                                        "answer": "वाराणसी में लगभग 88 घाट हैं जो गंगा नदी के किनारे स्थित हैं।"
                                    },
                                    {
                                        "id": "9",
                                        "question": "मणिकर्णिका घाट का क्या महत्व है?",
                                        "answer": "मणिकर्णिका घाट मुख्य श्मशान घाट है जहाँ हिंदू धर्म के अनुसार मोक्ष मिलता है।"
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "domain": "tourism",
                        "contexts": [
                            {
                                "context": "वाराणसी एक प्रमुख पर्यटन स्थल है जो अपनी संस्कृति और इतिहास के लिए जाना जाता है।",
                                "qas": [
                                    {
                                        "id": "10",
                                        "question": "वाराणसी जाने का सबसे अच्छा समय कब है?",
                                        "answer": "वाराणसी जाने का सबसे अच्छा समय अक्टूबर से मार्च तक है जब मौसम सुहावना होता है।"
                                    },
                                    {
                                        "id": "11",
                                        "question": "वाराणसी में कैसे पहुंचें?",
                                        "answer": "वाराणसी हवाई, रेल और सड़क मार्ग से अच्छी तरह जुड़ा है। लाल बहादुर शास्त्री हवाई अड्डा यहाँ का मुख्य हवाई अड्डा है।"
                                    },
                                    {
                                        "id": "12",
                                        "question": "वाराणसी में कहाँ रुकें?",
                                        "answer": "वाराणसी में गंगा के किनारे हेरिटेज होटल, गेस्ट हाउस और आधुनिक होटल उपलब्ध हैं।"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }

            # Save sample data
            with open("sample_vatika_data.json", "w", encoding="utf-8") as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)

            st.success("✅ Enhanced sample data created: sample_vatika_data.json")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 Chat", "🏋️ Training", "📊 Analytics", "🧪 Testing", "🔧 Advanced"])

    # Tab 1: Enhanced Chat Interface
    with tab1:
        st.header("🤖 Enhanced VATIKA Chat")

        if not model.trained:
            st.warning("⚠️ Model is not trained yet. Please train the model first in the Training tab.")
        else:
            # Chat input
            col1, col2 = st.columns([4, 1])
            with col1:
                user_query = st.text_input("🗣️ आपका प्रश्न (Your Question):",
                                         placeholder="जैसे: काशी विश्वनाथ मंदिर कहाँ है?")
            with col2:
                ask_button = st.button("पूछें", type="primary", use_container_width=True)

            # Process query
            if ask_button and user_query:
                with st.spinner("🔍 Searching for answer..."):
                    response, domain, confidence = chatbot.chat(user_query)

                # Display response
                st.markdown("### 🤖 VATIKA का उत्तर:")

                # Response with styling
                response_color = "success" if confidence > 0.5 else "warning" if confidence > 0.2 else "error"
                if response_color == "success":
                    st.markdown(f'<div class="success-box">{response}</div>', unsafe_allow_html=True)
                elif response_color == "warning":
                    st.markdown(f'<div class="warning-box">{response}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-box">{response}</div>', unsafe_allow_html=True)

                # Metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Domain", domain)
                with col2:
                    st.metric("Confidence", f"{confidence:.3f}")
                with col3:
                    confidence_percent = confidence * 100
                    st.metric("Confidence %", f"{confidence_percent:.1f}%")

            # Conversation History
            if st.checkbox("📜 Show Conversation History") and chatbot.conversation_history:
                st.markdown("### 📜 बातचीत का इतिहास")

                for i, conv in enumerate(reversed(chatbot.conversation_history[-5:])):  # Show last 5
                    with st.expander(f"Q{len(chatbot.conversation_history)-i}: {conv['query'][:50]}..."):
                        st.markdown(f"**🗣️ प्रश्न:** {conv['query']}")
                        st.markdown(f"**🤖 उत्तर:** {conv['answer']}")
                        st.markdown(f"**📂 डोमेन:** {conv['domain']}")
                        st.markdown(f"**📊 विश्वसनीयता:** {conv['confidence']:.3f}")
                        st.markdown(f"**⏰ समय:** {conv['timestamp']}")

    # Tab 2: Enhanced Training
    with tab2:
        st.header("🏋️ Enhanced Model Training")

        # Data Loading Section
        st.subheader("📂 Data Loading")

        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Choose JSON data file",
                type=['json'],
                help="Upload a JSON file with the VATIKA data format"
            )
        with col2:
            use_sample = st.checkbox("Use Sample Data",
                                   help="Use the built-in sample data for training")

        # Training Configuration
        st.subheader("⚙️ Training Configuration")

        col1, col2, col3 = st.columns(3)
        with col1:
            train_split = st.slider("Training Split", 0.5, 0.9, 0.7, 0.05)
        with col2:
            val_split = st.slider("Validation Split", 0.05, 0.3, 0.15, 0.05)
        with col3:
            test_split = 1.0 - train_split - val_split
            st.metric("Test Split", f"{test_split:.2f}")

        # Load and Train Button
        if st.button("🚀 Load Data & Train Model", type="primary"):
            data_file = None

            if use_sample:
                if os.path.exists("sample_vatika_data.json"):
                    data_file = "sample_vatika_data.json"
                else:
                    st.error("❌ Sample data not found. Please create sample data first.")
            elif uploaded_file:
                # Save uploaded file temporarily
                with open("temp_data.json", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                data_file = "temp_data.json"
            else:
                st.error("❌ Please upload a file or select sample data.")

            if data_file:
                # Load data with progress
                with st.spinner("📊 Loading and splitting data..."):
                    success, message = model.load_and_split_data(data_file, train_split, val_split)

                if success:
                    st.success(f"✅ {message}")

                    # Show data statistics
                    st.subheader("📊 Data Statistics")
                    domain_stats = model.get_domain_statistics()

                    # Create DataFrame for display
                    stats_df = pd.DataFrame(domain_stats).T
                    stats_df['Total'] = stats_df.sum(axis=1)
                    st.dataframe(stats_df)

                    # Training with progress bar
                    st.subheader("🏋️ Training Progress")

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def progress_callback(current, total, step_name):
                        progress = current / total
                        progress_bar.progress(progress)
                        status_text.text(f"Step {current}/{total}: {step_name}")

                    # Train model
                    train_success, train_message = model.train_model(progress_callback)

                    if train_success:
                        st.success("🎉 Training completed successfully!")
                        st.info(train_message)

                        # Display validation metrics
                        if model.validation_metrics:
                            st.subheader("📈 Validation Metrics")
                            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

                            with metrics_col1:
                                st.metric("Accuracy", f"{model.validation_metrics.get('accuracy', 0):.3f}")
                            with metrics_col2:
                                st.metric("F1 Score", f"{model.validation_metrics.get('f1', 0):.3f}")
                            with metrics_col3:
                                st.metric("Precision", f"{model.validation_metrics.get('precision', 0):.3f}")
                            with metrics_col4:
                                st.metric("Recall", f"{model.validation_metrics.get('recall', 0):.3f}")
                    else:
                        st.error(f"❌ Training failed: {train_message}")
                else:
                    st.error(f"❌ Data loading failed: {message}")

        # Load Pre-trained Model
        st.markdown("---")
        st.subheader("📥 Load Pre-trained Model")

        if st.button("📥 Load Existing Model"):
            with st.spinner("Loading pre-trained model..."):
                success, message = model.load_trained_model()

            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")

    # Tab 3: Enhanced Analytics
    with tab3:
        st.header("📊 Enhanced Model Analytics")

        if not model.trained:
            st.warning("⚠️ Model is not trained yet. Please train the model first.")
        else:
            # Domain Distribution
            st.subheader("🏷️ Domain Distribution")

            domain_stats = model.get_domain_statistics()
            if domain_stats:
                # Create visualization
                domains = list(domain_stats.keys())
                train_counts = [domain_stats[d]['train'] for d in domains]
                val_counts = [domain_stats[d]['val'] for d in domains]
                test_counts = [domain_stats[d]['test'] for d in domains]

                fig = go.Figure(data=[
                    go.Bar(name='Training', x=domains, y=train_counts),
                    go.Bar(name='Validation', x=domains, y=val_counts),
                    go.Bar(name='Test', x=domains, y=test_counts)
                ])

                fig.update_layout(
                    title="Data Distribution by Domain",
                    xaxis_title="Domains",
                    yaxis_title="Number of Entries",
                    barmode='group'
                )

                st.plotly_chart(fig, use_container_width=True)

            # Validation Metrics
            if model.validation_metrics:
                st.subheader("📈 Model Performance")

                metrics_df = pd.DataFrame([model.validation_metrics])
                st.dataframe(metrics_df, use_container_width=True)

                # Metrics visualization
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Accuracy', 'F1 Score', 'Precision', 'Recall'],
                        y=[
                            model.validation_metrics.get('accuracy', 0),
                            model.validation_metrics.get('f1', 0),
                            model.validation_metrics.get('precision', 0),
                            model.validation_metrics.get('recall', 0)
                        ]
                    )
                ])

                fig.update_layout(
                    title="Validation Metrics",
                    yaxis_title="Score",
                    yaxis=dict(range=[0, 1])
                )

                st.plotly_chart(fig, use_container_width=True)

            # Sample Predictions Analysis
            st.subheader("🔍 Sample Predictions Analysis")

            if st.button("🧪 Analyze Sample Predictions"):
                if model.val_data:
                    sample_size = min(5, len(model.val_data))
                    sample_entries = np.random.choice(model.val_data, sample_size, replace=False)

                    for i, entry in enumerate(sample_entries):
                        with st.expander(f"Sample {i+1}: {entry.question[:50]}..."):
                            st.markdown(f"**Original Question:** {entry.question}")
                            st.markdown(f"**Original Answer:** {entry.answer}")
                            st.markdown(f"**Domain:** {entry.domain}")

                            # Get predictions
                            predictions = model.predict_single(entry.question)

                            if predictions:
                                st.markdown("**Top Predictions:**")
                                for j, (pred, conf) in enumerate(predictions[:3]):
                                    st.markdown(f"{j+1}. **{pred.answer}** (Confidence: {conf:.3f}, Domain: {pred.domain})")

    # Tab 4: Enhanced Testing
    with tab4:
        st.header("🧪 Enhanced Model Testing")

        if not model.trained:
            st.warning("⚠️ Model is not trained yet. Please train the model first.")
        else:
            # Interactive Testing
            st.subheader("🔧 Interactive Testing")

            test_query = st.text_area(
                "Enter test query:",
                placeholder="Enter a question to test the model's response"
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                top_k = st.slider("Number of predictions to show:", 1, 10, 3)
            with col2:
                exclude_domains = st.multiselect(
                    "Exclude domains:",
                    options=list(set(entry.domain for entry in model.train_data)) if model.train_data else []
                )

            if st.button("🧪 Test Query") and test_query:
                with st.spinner("Testing query..."):
                    predictions = model.predict_single(test_query, exclude_domains)

                if predictions:
                    st.markdown("### 🎯 Predictions:")

                    for i, (pred, confidence) in enumerate(predictions[:top_k]):
                        with st.expander(f"Prediction {i+1} (Confidence: {confidence:.3f})"):
                            st.markdown(f"**Answer:** {pred.answer}")
                            st.markdown(f"**Domain:** {pred.domain}")
                            st.markdown(f"**Context:** {pred.context}")
                            st.markdown(f"**Original Question:** {pred.question}")

                            # Confidence color coding
                            if confidence > 0.5:
                                st.success(f"High confidence: {confidence:.3f}")
                            elif confidence > 0.2:
                                st.warning(f"Medium confidence: {confidence:.3f}")
                            else:
                                st.error(f"Low confidence: {confidence:.3f}")
                else:
                    st.error("No predictions found for this query.")

            # Batch Testing
            st.markdown("---")
            st.subheader("📊 Batch Testing")

            if st.button("🏃 Run Test Set Evaluation"):
                if model.test_data:
                    with st.spinner("Running test set evaluation..."):
                        test_results = model.test_model()

                    if test_results:
                        st.subheader("📈 Test Results")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Test Accuracy", f"{test_results.get('test_accuracy', 0):.3f}")
                        with col2:
                            st.metric("Correct Predictions", test_results.get('test_correct', 0))
                        with col3:
                            st.metric("Total Test Cases", test_results.get('test_total', 0))
                else:
                    st.warning("No test data available.")

    # Tab 5: Advanced Features
    with tab5:
        st.header("🔧 Advanced Features")

        # Model Information
        st.subheader("ℹ️ Model Information")

        if model.trained:
            model_info = {
                "Training Entries": len(model.train_data),
                "Validation Entries": len(model.val_data),
                "Test Entries": len(model.test_data),
                "Unique Domains": len(set(entry.domain for entry in model.train_data)) if model.train_data else 0,
                "Question Vectorizer Features": model.question_vectorizer.max_features if hasattr(model.question_vectorizer, 'max_features') else "N/A",
                "Context Vectorizer Features": model.context_vectorizer.max_features if hasattr(model.context_vectorizer, 'max_features') else "N/A",
                "Domain Classifier": "Available" if model.domain_classifier else "Not Available"
            }

            info_df = pd.DataFrame(list(model_info.items()), columns=['Property', 'Value'])
            st.dataframe(info_df, use_container_width=True)

        # Export Options
        st.subheader("📤 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Export Training Data"):
                if model.train_data:
                    export_data = [entry.to_dict() for entry in model.train_data]
                    st.download_button(
                        label="Download Training Data",
                        data=json.dumps(export_data, ensure_ascii=False, indent=2),
                        file_name="training_data.json",
                        mime="application/json"
                    )

        with col2:
            if st.button("📈 Export Metrics"):
                if model.validation_metrics:
                    metrics_json = json.dumps(model.validation_metrics, indent=2)
                    st.download_button(
                        label="Download Metrics",
                        data=metrics_json,
                        file_name="model_metrics.json",
                        mime="application/json"
                    )

        with col3:
            if st.button("💾 Export Model Info"):
                if model.trained:
                    model_info_data = {
                        "training_size": len(model.train_data),
                        "validation_size": len(model.val_data),
                        "test_size": len(model.test_data),
                        "domains": list(set(entry.domain for entry in model.train_data)) if model.train_data else [],
                        "validation_metrics": model.validation_metrics,
                        "export_timestamp": datetime.now().isoformat()
                    }

                    st.download_button(
                        label="Download Model Info",
                        data=json.dumps(model_info_data, ensure_ascii=False, indent=2),
                        file_name="model_info.json",
                        mime="application/json"
                    )

        # Clear Data
        st.markdown("---")
        st.subheader("🗑️ Data Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Reset Model", type="secondary"):
                if st.button("⚠️ Confirm Reset"):
                    st.session_state.enhanced_model = EnhancedVATIKAModel()
                    st.session_state.enhanced_chatbot = EnhancedVATIKAChatbot(st.session_state.enhanced_model)
                    st.success("✅ Model reset successfully!")
                    st.experimental_rerun()

        with col2:
            if st.button("🧹 Clear Chat History"):
                if hasattr(st.session_state, 'enhanced_chatbot'):
                    st.session_state.enhanced_chatbot.conversation_history = []
                    st.success("✅ Chat history cleared!")

        # Debug Information
        if st.checkbox("🐛 Show Debug Information"):
            st.subheader("🐛 Debug Information")

            debug_info = {
                "Session State Keys": list(st.session_state.keys()),
                "Model Trained": model.trained,
                "Model Directory": model.model_dir,
                "Model Directory Exists": os.path.exists(model.model_dir) if hasattr(model, 'model_dir') else False,
            }

            st.json(debug_info)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        🕉️ VATIKA - Enhanced ML Pipeline for Varanasi Tourism<br>
        Advanced Question Answering System with Machine Learning
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()