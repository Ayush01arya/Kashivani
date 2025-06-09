from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)


class EnhancedQAChatbot:
    """Enhanced QA Chatbot class for Hindi questions"""

    def __init__(self):
        self.qa_pairs = []
        self.vectorizer = None
        self.qa_vectors = None
        self.validation_qa_pairs = []
        self.model_loaded = False

    def load_model(self, model_path):
        """Load the trained model from pickle file"""
        try:
            if not os.path.exists(model_path):
                print(f"❌ Model file '{model_path}' not found!")
                return False

            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Load all components
            self.qa_pairs = model_data.get('qa_pairs', [])
            self.vectorizer = model_data.get('vectorizer')
            self.qa_vectors = model_data.get('qa_vectors')
            self.validation_qa_pairs = model_data.get('validation_qa_pairs', [])
            self.model_loaded = True

            print(f"✅ Model loaded: {len(self.qa_pairs)} QA pairs")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def generate_answer(self, question):
        """Generate answer for a given question"""
        if not self.model_loaded or not self.vectorizer or self.qa_vectors is None:
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

    def get_stats(self):
        """Get model statistics"""
        return {
            "training_pairs": len(self.qa_pairs),
            "validation_pairs": len(self.validation_qa_pairs),
            "domains_covered": len(set([qa.get('domain', 'unknown') for qa in self.qa_pairs])),
            "model_loaded": self.model_loaded
        }


# Initialize chatbot globally
chatbot = EnhancedQAChatbot()

# Load model on startup
MODEL_PATH = 'optimized_hindi_qa_model.pkl'
chatbot.load_model(MODEL_PATH)

# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="hi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hindi QA Chatbot API</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
            text-align: center;
            padding: 30px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .content {
            padding: 30px;
        }
        .input-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #333;
        }
        input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            box-sizing: border-box;
        }
        input[type="text"]:focus {
            border-color: #667eea;
            outline: none;
        }
        button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            margin-right: 10px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .response {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            min-height: 50px;
        }
        .api-info {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .endpoint {
            background: #1a1a1a;
            color: #00ff00;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            margin: 5px 0;
        }
        .status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        .status.online {
            background: #d4edda;
            color: #155724;
        }
        .status.offline {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Hindi QA Chatbot API</h1>
            <p>AI-powered Hindi Question Answering System</p>
            <span class="status {{ 'online' if model_loaded else 'offline' }}">
                {{ 'Model Loaded ✅' if model_loaded else 'Model Not Loaded ❌' }}
            </span>
        </div>

        <div class="content">
            <div class="input-group">
                <label for="question">प्रश्न पूछें (Ask Question in Hindi):</label>
                <input type="text" id="question" placeholder="यहाँ अपना प्रश्न लिखें..." />
            </div>

            <button onclick="askQuestion()">प्रश्न पूछें</button>
            <button onclick="getStats()">Model Stats</button>

            <div class="response" id="response">
                यहाँ उत्तर दिखाई देगा...
            </div>

            <div class="api-info">
                <h3>🔗 API Endpoints</h3>
                <p><strong>Ask Question:</strong></p>
                <div class="endpoint">POST /api/ask</div>
                <p>Body: {"question": "आपका प्रश्न"}</p>

                <p><strong>Health Check:</strong></p>
                <div class="endpoint">GET /api/health</div>

                <p><strong>Model Stats:</strong></p>
                <div class="endpoint">GET /api/stats</div>
            </div>
        </div>
    </div>

    <script>
        async function askQuestion() {
            const question = document.getElementById('question').value;
            const responseDiv = document.getElementById('response');

            if (!question.trim()) {
                responseDiv.innerHTML = '⚠️ कृपया प्रश्न लिखें!';
                return;
            }

            responseDiv.innerHTML = '🔄 उत्तर खोजा जा रहा है...';

            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: question })
                });

                const data = await response.json();

                if (data.success) {
                    responseDiv.innerHTML = `
                        <strong>🤖 उत्तर:</strong><br>
                        ${data.answer}<br><br>
                        <small>Confidence: ${(data.confidence * 100).toFixed(1)}%</small>
                    `;
                } else {
                    responseDiv.innerHTML = `❌ ${data.error}`;
                }
            } catch (error) {
                responseDiv.innerHTML = `❌ Error: ${error.message}`;
            }
        }

        async function getStats() {
            const responseDiv = document.getElementById('response');
            responseDiv.innerHTML = '📊 Stats लोड हो रहे हैं...';

            try {
                const response = await fetch('/api/stats');
                const data = await response.json();

                responseDiv.innerHTML = `
                    <strong>📊 Model Statistics:</strong><br>
                    • Training QA pairs: ${data.training_pairs}<br>
                    • Validation QA pairs: ${data.validation_pairs}<br>
                    • Domains covered: ${data.domains_covered}<br>
                    • Model Status: ${data.model_loaded ? 'Loaded ✅' : 'Not Loaded ❌'}
                `;
            } catch (error) {
                responseDiv.innerHTML = `❌ Error: ${error.message}`;
            }
        }

        // Allow Enter key to submit
        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    """Home page with web interface"""
    return render_template_string(HTML_TEMPLATE, model_loaded=chatbot.model_loaded)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': chatbot.model_loaded,
        'message': 'Hindi QA Chatbot API is running'
    })


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Main endpoint to ask questions"""
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'Question is required'
            }), 400

        question = data['question'].strip()

        if not question:
            return jsonify({
                'success': False,
                'error': 'Question cannot be empty'
            }), 400

        # Generate answer
        answer = chatbot.generate_answer(question)

        # Calculate confidence (basic implementation)
        confidence = 0.8 if "माफ करें" not in answer else 0.1

        return jsonify({
            'success': True,
            'question': question,
            'answer': answer,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    try:
        stats = chatbot.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/batch', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple questions"""
    try:
        data = request.get_json()

        if not data or 'questions' not in data:
            return jsonify({
                'success': False,
                'error': 'Questions list is required'
            }), 400

        questions = data['questions']

        if not isinstance(questions, list):
            return jsonify({
                'success': False,
                'error': 'Questions must be a list'
            }), 400

        results = []

        for question in questions:
            if isinstance(question, str) and question.strip():
                answer = chatbot.generate_answer(question.strip())
                confidence = 0.8 if "माफ करें" not in answer else 0.1

                results.append({
                    'question': question.strip(),
                    'answer': answer,
                    'confidence': confidence
                })
            else:
                results.append({
                    'question': question,
                    'answer': 'Invalid question format',
                    'confidence': 0.0
                })

        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)