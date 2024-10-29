from flask import Flask, jsonify, request, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

# Параметри підключення до бази даних
DB_USER = 'SergNik'
DB_PASSWORD = 'Sergiy2024'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'sample30'

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Модель для таблиці sensors_data
class SensorData(db.Model):
    __tablename__ = 'sensors_data'
    data_id = db.Column(db.Integer, primary_key=True)
    sensor_id = db.Column(db.Integer, nullable=False)
    temperature = db.Column(db.Numeric(5, 2))
    humidity = db.Column(db.Numeric(5, 2))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Ініціалізація бази даних
@app.before_first_request
def create_tables():
    db.create_all()

# Отримати всі записи з датчиків
@app.route('/api/sensors', methods=['GET'])
def get_all_sensor_data():
    data = SensorData.query.all()
    result = [{"data_id": d.data_id, "sensor_id": d.sensor_id, "temperature": str(d.temperature),
               "humidity": str(d.humidity), "timestamp": d.timestamp} for d in data]
    return jsonify(result)

# Додати новий запис з датчика
@app.route('/api/sensors', methods=['POST'])
def add_sensor_data():
    data = request.get_json()
    new_data = SensorData(sensor_id=data['sensor_id'], temperature=data['temperature'], humidity=data['humidity'])
    db.session.add(new_data)
    db.session.commit()
    return jsonify({"message": "New sensor data added"}), 201

# Отримати дані для конкретного датчика
@app.route('/api/sensors/<int:sensor_id>', methods=['GET'])
def get_sensor_data_by_id(sensor_id):
    data = SensorData.query.filter_by(sensor_id=sensor_id).all()
    result = [{"data_id": d.data_id, "sensor_id": d.sensor_id, "temperature": str(d.temperature),
               "humidity": str(d.humidity), "timestamp": d.timestamp} for d in data]
    return jsonify(result)

# Видалити запис за data_id
@app.route('/api/sensors/<int:data_id>', methods=['DELETE'])
def delete_sensor_data(data_id):
    data = SensorData.query.get(data_id)
    if data:
        db.session.delete(data)
        db.session.commit()
        return jsonify({"message": "Sensor data deleted"}), 200
    return jsonify({"message": "Data not found"}), 404

# Головна сторінка
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


