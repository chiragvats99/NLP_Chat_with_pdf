import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
import spacy
from transformers import pipeline
from flask import Flask, render_template, request, jsonify

# Download NLTK resources
nltk.download('punkt')

# Download spaCy model
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Initialize Flask app
app = Flask(__name__)

# Load BERT-based question answering model
qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Define routes and functions
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})
    
    file = request.files['file']

    # Extract text from PDF
    pdf_text = extract_text_from_pdf(file)

    # Perform NLP analysis
    sentences = sent_tokenize(pdf_text)
    entities = extract_entities(pdf_text)

    # Generate probable questions
    probable_questions = generate_questions(sentences, entities)

    return render_template('result.html', text=pdf_text, questions=probable_questions)

@app.route('/answer', methods=['POST'])
def answer():
    question = request.form.get('question')

    # Use BERT for question answering
    answer = qa_model(context=request.form.get('text'), question=question)

    return jsonify({'answer': answer['answer']})

# Helper functions
def extract_text_from_pdf(file):
    pdf_document = fitz.open(file)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()

    return text

def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def generate_questions(sentences, entities):
    def generate_questions(sentences, entities):
    probable_questions = []

    for sentence in sentences:
        # Tokenize the sentence
        sentence_tokens = nlp(sentence)

        # Check if the sentence contains an entity
        if any(entity in sentence for entity in entities):
            # Generate a "What" question for the sentence
            what_question = f"What is {sentence_tokens[entities[0]].lemma_}?"
            probable_questions.append(what_question)

            # Generate a "Who" question for the sentence
            who_question = f"Who {sentence_tokens[entities[0]].lemma_}?"
            probable_questions.append(who_question)

            # You can add more question types based on your requirements

    return probable_questions


if __name__ == "__main__":
        app.run(debug=True)
