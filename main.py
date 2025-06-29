import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM, pipeline,
    Trainer, TrainingArguments
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import re
import logging
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HindiQAModel:
    def __init__(self, model_name="ai4bharat/IndicBARTSS"):
        """
        Initialize the Hindi QA model with open-source LLM
        Using IndicBART for better Hindi language support
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.qa_pipeline = None

    def load_model(self):
        """Load the pre-trained model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

            # Create QA pipeline
            self.qa_pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to a simpler model
            self.model_name = "google/mt5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.qa_pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

    def load_data(self, file_path: str) -> List[Dict]:
        """Load data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self.parse_data(data)
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return []

    def parse_data(self, data: Dict) -> List[Dict]:
        """Parse the nested JSON structure into flat list"""
        parsed_data = []

        for domain in data.get('domains', []):
            domain_name = domain.get('domain', '')

            for context_item in domain.get('contexts', []):
                context = context_item.get('context', '')

                for qa in context_item.get('qas', []):
                    parsed_data.append({
                        'id': qa.get('id', ''),
                        'domain': domain_name,
                        'context': context,
                        'question': qa.get('question', ''),
                        'answer': qa.get('answer', '').strip()
                    })

        return parsed_data

    def preprocess_for_training(self, data: List[Dict]) -> Dataset:
        """Preprocess data for training"""
        inputs = []
        targets = []

        for item in data:
            # Create input prompt for seq2seq model
            input_text = f"प्रसंग: {item['context']}\nप्रश्न: {item['question']}\nउत्तर:"
            target_text = item['answer']

            inputs.append(input_text)
            targets.append(target_text)

        # Tokenize
        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        dataset_dict = {
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'labels': labels['input_ids']
        }

        return Dataset.from_dict(dataset_dict)

    def train_model(self, train_data: List[Dict], val_data: List[Dict]):
        """Fine-tune the model on training data"""
        logger.info("Preparing training data...")
        train_dataset = self.preprocess_for_training(train_data)
        val_dataset = self.preprocess_for_training(val_data)

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./hindi_qa_model',
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )

        # Train the model
        logger.info("Starting training...")
        trainer.train()

        # Save the model
        trainer.save_model('./hindi_qa_model_final')
        self.tokenizer.save_pretrained('./hindi_qa_model_final')
        logger.info("Training completed and model saved!")

    def predict_answer(self, context: str, question: str) -> str:
        """Generate answer for a given context and question"""
        try:
            # Create input prompt
            input_text = f"प्रसंग: {context}\nप्रश्न: {question}\nउत्तर:"

            # Generate answer
            result = self.qa_pipeline(
                input_text,
                max_length=150,
                num_beams=4,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

            answer = result[0]['generated_text']

            # Clean the answer (remove the input prompt if it's repeated)
            if "उत्तर:" in answer:
                answer = answer.split("उत्तर:")[-1].strip()

            return answer

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return "उत्तर उत्पन्न करने में त्रुटि।"

    def predict_test_data(self, test_data: List[Dict]) -> List[Dict]:
        """Generate predictions for test data"""
        logger.info("Generating predictions for test data...")
        predictions = []

        for item in test_data:
            predicted_answer = self.predict_answer(item['context'], item['question'])

            prediction = {
                'id': item['id'],
                'domain': item['domain'],
                'context': item['context'],
                'question': item['question'],
                'answer': predicted_answer
            }
            predictions.append(prediction)

        return predictions

    def calculate_f1_score(self, predicted: List[str], actual: List[str]) -> float:
        """Calculate F1 score at word level"""
        f1_scores = []

        for pred, actual_ans in zip(predicted, actual):
            pred_words = set(pred.split())
            actual_words = set(actual_ans.split())

            if len(pred_words) == 0 and len(actual_words) == 0:
                f1_scores.append(1.0)
            elif len(pred_words) == 0 or len(actual_words) == 0:
                f1_scores.append(0.0)
            else:
                common = pred_words.intersection(actual_words)
                precision = len(common) / len(pred_words)
                recall = len(common) / len(actual_words)

                if precision + recall == 0:
                    f1_scores.append(0.0)
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    f1_scores.append(f1)

        return np.mean(f1_scores)

    def calculate_bleu_score(self, predicted: List[str], actual: List[str]) -> float:
        """Calculate BLEU score"""
        bleu_scores = []
        smoothing = SmoothingFunction().method1

        for pred, actual_ans in zip(predicted, actual):
            pred_tokens = pred.split()
            actual_tokens = [actual_ans.split()]

            if len(pred_tokens) == 0:
                bleu_scores.append(0.0)
            else:
                bleu = sentence_bleu(actual_tokens, pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(bleu)

        return np.mean(bleu_scores)

    def calculate_rouge_score(self, predicted: List[str], actual: List[str]) -> float:
        """Calculate ROUGE-L score"""
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        rouge_scores = []

        for pred, actual_ans in zip(predicted, actual):
            scores = scorer.score(actual_ans, pred)
            rouge_scores.append(scores['rougeL'].fmeasure)

        return np.mean(rouge_scores)

    def evaluate_model(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Evaluate model performance"""
        # Extract predictions and ground truth
        pred_answers = [p['answer'] for p in predictions]
        true_answers = [gt['answer'] for gt in ground_truth]

        # Calculate metrics
        f1 = self.calculate_f1_score(pred_answers, true_answers)
        bleu = self.calculate_bleu_score(pred_answers, true_answers)
        rouge_l = self.calculate_rouge_score(pred_answers, true_answers)

        metrics = {
            'F1_Score': f1,
            'BLEU_Score': bleu,
            'ROUGE-L_Score': rouge_l
        }

        return metrics

    def save_predictions_to_json(self, predictions: List[Dict], output_file: str):
        """Save predictions in the required JSON format"""
        # Group predictions by domain
        domains_dict = {}

        for pred in predictions:
            domain = pred['domain']
            if domain not in domains_dict:
                domains_dict[domain] = {}

            context = pred['context']
            if context not in domains_dict[domain]:
                domains_dict[domain][context] = []

            qa_item = {
                'id': pred['id'],
                'question': pred['question'],
                'answer': pred['answer']
            }
            domains_dict[domain][context].append(qa_item)

        # Convert to required format
        output_data = {'domains': []}

        for domain_name, contexts in domains_dict.items():
            domain_data = {
                'domain': domain_name,
                'contexts': []
            }

            for context, qas in contexts.items():
                context_data = {
                    'context': context,
                    'qas': qas
                }
                domain_data['contexts'].append(context_data)

            output_data['domains'].append(domain_data)

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Predictions saved to {output_file}")


def main():
    """Main execution function"""
    # Initialize the model
    qa_model = HindiQAModel()
    qa_model.load_model()

    # Load training and validation data
    logger.info("Loading training data...")
    train_data = qa_model.load_data('train.json')

    logger.info("Loading validation data...")
    val_data = qa_model.load_data('validation.json')

    # Train the model (uncomment to train)
    # qa_model.train_model(train_data, val_data)

    # For demonstration, we'll use the pre-trained model for predictions
    # Load test data (you would load Test Data-II here)
    logger.info("Loading test data...")
    # test_data = qa_model.load_data('test_data_2.json')  # Replace with actual test file

    # For demo, using validation data as test data
    test_data = val_data[:10]  # Taking first 10 samples for demo

    # Generate predictions
    predictions = qa_model.predict_test_data(test_data)

    # Save predictions
    qa_model.save_predictions_to_json(predictions, 'predictions.json')

    # Evaluate (if ground truth is available)
    if val_data:
        metrics = qa_model.evaluate_model(predictions, test_data)
        logger.info("Evaluation Metrics:")
        for metric, score in metrics.items():
            logger.info(f"{metric}: {score:.4f}")

    logger.info("Process completed successfully!")


if __name__ == "__main__":
    main()