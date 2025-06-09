# Test your trained model
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class EnhancedQAChatbot:
    """Enhanced QA Chatbot class for Hindi questions"""

    def __init__(self):
        self.qa_pairs = []
        self.vectorizer = None
        self.qa_vectors = None
        self.validation_qa_pairs = []

    def load_model(self, model_path):
        """Load the trained model from pickle file"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Load all components
            self.qa_pairs = model_data.get('qa_pairs', [])
            self.vectorizer = model_data.get('vectorizer')
            self.qa_vectors = model_data.get('qa_vectors')
            self.validation_qa_pairs = model_data.get('validation_qa_pairs', [])

            print(f"✅ Model loaded: {len(self.qa_pairs)} QA pairs")
            return True

        except FileNotFoundError:
            print(f"❌ Model file '{model_path}' not found!")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def generate_answer(self, question):
        """Generate answer for a given question"""
        if not self.vectorizer or self.qa_vectors is None:
            return "मॉडल लोड नहीं है। कृपया पहले मॉडल लोड करें।"

        try:
            # Vectorize the question
            question_vector = self.vectorizer.transform([question])

            # Calculate similarities
            similarities = cosine_similarity(question_vector, self.qa_vectors).flatten()

            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]

            # Return answer if similarity is above threshold
            if best_similarity > 0.1:  # Adjust threshold as needed
                return self.qa_pairs[best_idx]['answer']
            else:
                return "माफ करें, मैं इस प्रश्न का उत्तर नहीं दे सकता।"

        except Exception as e:
            return f"Error generating answer: {e}"

    def generate_answer_optimized(self, question):
        """Optimized version of answer generation (if available)"""
        # This is the same as generate_answer for now
        # You can implement optimization here if needed
        return self.generate_answer(question)


# Load your saved model
def load_and_test_model():
    print("Loading trained model...")

    # Create a new chatbot instance
    chatbot = EnhancedQAChatbot()

    # Load the saved model
    if not chatbot.load_model('optimized_hindi_qa_model.pkl'):
        print("❌ Failed to load model. Please check if 'optimized_hindi_qa_model.pkl' exists.")
        return

    print("Model loaded successfully!")

    # Interactive testing
    print("\n" + "=" * 50)
    print("🤖 Hindi QA Chatbot Ready!")
    print("Type your questions in Hindi")
    print("Type 'quit' to exit")
    print("Type 'stats' to see model statistics")
    print("=" * 50)

    while True:
        user_query = input("\n📝 प्रश्न पूछें: ")

        if user_query.lower() == 'quit':
            print("👋 धन्यवाद!")
            break

        elif user_query.lower() == 'stats':
            print(f"📊 Model Statistics:")
            print(f"   • Training QA pairs: {len(chatbot.qa_pairs)}")
            print(
                f"   • Validation QA pairs: {len(chatbot.validation_qa_pairs) if hasattr(chatbot, 'validation_qa_pairs') else 'N/A'}")
            print(f"   • Domains covered: {len(set([qa['domain'] for qa in chatbot.qa_pairs if 'domain' in qa]))}")
            continue

        # Get answer
        answer = chatbot.generate_answer_optimized(user_query) if hasattr(chatbot,
                                                                          'generate_answer_optimized') else chatbot.generate_answer(
            user_query)
        print(f"🤖 उत्तर: {answer}")


# Test specific questions
def test_sample_questions():
    print("Testing with sample questions...")

    chatbot = EnhancedQAChatbot()
    if not chatbot.load_model('optimized_hindi_qa_model.pkl'):
        print("❌ Failed to load model for testing.")
        return

    # Sample questions (modify based on your data)
    sample_questions = [
        "कुंड कहाँ स्थित है?",
        "दर्शन का समय क्या है?",
        "कैसे पहुंचा जा सकता है?",
        "दूरी कितनी है?",
        "क्या विशेषता है?"
    ]

    print("\n📋 Sample Test Results:")
    print("-" * 60)

    for i, question in enumerate(sample_questions, 1):
        answer = chatbot.generate_answer_optimized(question) if hasattr(chatbot,
                                                                        'generate_answer_optimized') else chatbot.generate_answer(
            question)
        print(f"{i}. प्रश्न: {question}")
        print(f"   उत्तर: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        print()


# Prepare for Test Data-II
def prepare_for_submission():
    """Prepare function for final test data submission"""

    def predict_test_data_ii(test_file_path, output_file_path):
        print(f"Loading test data from {test_file_path}...")

        # Load trained model
        chatbot = EnhancedQAChatbot()
        if not chatbot.load_model('optimized_hindi_qa_model.pkl'):
            print("❌ Failed to load model for prediction.")
            return None

        # Load test data
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Test file '{test_file_path}' not found!")
            return None

        prediction_count = 0

        # Generate predictions
        for domain_data in test_data['domains']:
            for context_data in domain_data['contexts']:
                for qa in context_data['qas']:
                    question = qa['question']

                    # Generate prediction
                    if hasattr(chatbot, 'generate_answer_optimized'):
                        predicted_answer = chatbot.generate_answer_optimized(question)
                    else:
                        predicted_answer = chatbot.generate_answer(question)

                    # Fill in the answer
                    qa['answer'] = predicted_answer
                    prediction_count += 1

        # Save predictions
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        print(f"✅ {prediction_count} predictions generated!")
        print(f"📁 Results saved to: {output_file_path}")

        return test_data

    return predict_test_data_ii


if __name__ == "__main__":
    print("🚀 What would you like to do?")
    print("1. Interactive testing")
    print("2. Sample question testing")
    print("3. Both")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        load_and_test_model()
    elif choice == "2":
        test_sample_questions()
    elif choice == "3":
        test_sample_questions()
        print("\n" + "=" * 50)
        load_and_test_model()
    else:
        print("Invalid choice!")

# For final submission (when you get Test Data-II)
"""
WHEN YOU GET TEST DATA-II FILE:

1. Place the test file (e.g., 'test_data_2.json') in your project folder
2. Run this code:

predict_function = prepare_for_submission()
predict_function('test_data_2.json', 'final_submission.json')

This will generate your final submission file!
"""