# SCRIPT PARA HACER PRIMERAS PRUEBAS DE MENSAJERIA
from twilio.rest import Client
from flask import Flask, request, jsonify

app = Flask(__name__)

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

DISEASES = ['minador', 'alternaria']

def send_message(disease):
    message = f'¡Alerta! Se ha detectado la enfermedad: {disease}'
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=TARGET_PHONE_NUMBER
    )

@app.route('/twilio', methods=['POST'])
def receive_prediction():
    data = request.json
    if 'prediction' in data:
        prediction = data['prediction'].lower()

        if prediction in DISEASES:
            send_message(prediction)
            return jsonify({'status': f'Mensaje enviado: {prediction} detectada'}), 200
        else:
            return jsonify({'status': 'No se detectó ninguna enfermedad'}), 200
    return jsonify({'error': 'Falta el campo "prediction"'}), 400

if __name__ == '__main__':
    app.run(debug=True)
