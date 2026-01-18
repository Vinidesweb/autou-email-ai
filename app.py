from flask import Flask, render_template, request
import os
import re
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=os.getenv("HF_TOKEN")
)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zà-ú0-9\s]', '', text)
    return text

def classify_email(email_text):
    response = client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um assistente que classifica emails.\n\n"
                    "Responda EXATAMENTE neste formato:\n"
                    "Classe: Produtivo OU Improdutivo\n"
                    "Resposta: <resposta curta e profissional>"
                )
            },
            {
                "role": "user",
                "content": email_text
            }
        ],
        max_tokens=200
    )

    text = response.choices[0].message.content


    classe = ""
    resposta = ""

    for line in text.splitlines():
        if line.lower().startswith("classe"):
            classe = line.split(":", 1)[1].strip()
        if line.lower().startswith("resposta"):
            resposta = line.split(":", 1)[1].strip()

    return {
        "classe": classe,
        "resposta": resposta
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()
        email_file = request.files.get("email_file")

        # Caso 1: texto colado
        if email_text:
            content = email_text

        # Caso 2: arquivo .txt
        elif email_file and email_file.filename.endswith(".txt"):
            content = email_file.read().decode("utf-8")

        else:
            content = None

        if content:
            result = classify_email(content)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
