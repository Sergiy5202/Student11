from flask import Flask, jsonify, request, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

# Параметри підключення до бази даних
DB_USER = 'postgres'
DB_PASSWORD = 'Sergiy2024'
DB_HOST = 'localhost'
DB_PORT = '5432'  # Вказуємо порт
DB_NAME = 'sample10'

app = Flask(__name__)
CORS(app)

# Конфігурація бази даних з використанням порту 5434
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgres'//{'Sergiy2024'}@'localhost'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Модель для таблиці датчиків (вона вже створена в базі даних)
class Sensor(db.Model):
    __tablename__ = 'sensors'
    
    sensor_id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(100), nullable=False)
    status = db.Column(db.Boolean, default=True)

# API: Отримати всі датчики
@app.route('/api/sensors', methods=['GET'])
def get_sensors():
    sensors = Sensor.query.all()
    result = [{'sensor_id': s.sensor_id, 'location': s.location, 'type': s.type, 'status': s.status} for s in sensors]
    return jsonify(result)

# API: Додати новий датчик
@app.route('/api/sensors', methods=['POST'])
def add_sensor():
    data = request.json
    new_sensor = Sensor(location=data['location'], type=data['type'], status=data['status'])
    db.session.add(new_sensor)
    db.session.commit()
    return jsonify({'message': 'Sensor added successfully'}), 201

# API: Видалити датчик за sensor_id
@app.route('/api/sensors/<int:sensor_id>', methods=['DELETE'])
def delete_sensor(sensor_id):
    sensor = Sensor.query.get(sensor_id)
    if not sensor:
        return jsonify({'message': 'Sensor not found'}), 404
    db.session.delete(sensor)
    db.session.commit()
    return jsonify({'message': 'Sensor deleted successfully'})

# API: Оновити інформацію про датчик
@app.route('/api/sensors/<int:sensor_id>', methods=['PUT'])
def update_sensor(sensor_id):
    sensor = Sensor.query.get(sensor_id)
    if not sensor:
        return jsonify({'message': 'Sensor not found'}), 404
    
    data = request.json
    sensor.location = data.get('location', sensor.location)
    sensor.type = data.get('type', sensor.type)
    sensor.status = data.get('status', sensor.status)
    db.session.commit()
    
    return jsonify({'message': 'Sensor updated successfully'})

# Головна сторінка
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
