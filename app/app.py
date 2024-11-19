from flask import Flask, request, jsonify
import spacy

# Carregar o modelo treinado
MODEL_PATH = "../models/chatbot_spacy_model"
nlp = spacy.load(MODEL_PATH)

# Iniciar a aplicação Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "Chatbot spaCy API está funcionando!"

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "Nenhuma mensagem fornecida"}), 400

    doc = nlp(user_input)
    predictions = doc.cats  # Categorias previstas pelo modelo
    intent = max(predictions, key=predictions.get)  # Intenção com maior probabilidade

    return jsonify({"intention": intent, "confidence": predictions[intent]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
