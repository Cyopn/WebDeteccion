from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# Ruta que recibe texto y lo devuelve en MAYÚSCULAS
@app.route('/mayusculas', methods=['POST'])
def a_mayusculas():
    data = request.get_json()
    texto = data.get("texto", "")
    return jsonify({"resultado": texto.upper()})

# Ruta que recibe texto y lo devuelve en minúsculas
@app.route('/minusculas', methods=['POST'])
def a_minusculas():
    data = request.get_json()
    texto = data.get("texto", "")
    return jsonify({"resultado": texto.lower()})

if __name__ == '__main__':
    app.run(debug=True, port=5501,host='127.0.0.1')