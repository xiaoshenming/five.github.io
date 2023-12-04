from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    classifier = pipeline("text-classification", model="your_pretrained_model", tokenizer="your_pretrained_model")
    result = classifier(text)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
