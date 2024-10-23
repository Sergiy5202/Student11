from adm-1 import Flask, jsonify, request
from adm-1 import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:password@localhost/practice_db'
db = SQLAlchemy(app)

class Task(db.Model):
    task_id = db.Column(db.Integer, primary_key=True)
    task_name = db.Column(db.String(100), nullable=False)
    due_date = db.Column(db.Date, nullable=False)

# Отримати всі завдання
@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    tasks = Task.query.all()
    return jsonify([{'task_id': task.task_id, 'task_name': task.task_name, 'due_date': str(task.due_date)} for task in tasks])

# Додати нове завдання
@app.route('/api/tasks', methods=['POST'])
def add_task():
    data = request.json
    new_task = Task(task_name=data['task_name'], due_date=data['due_date'])
    db.session.add(new_task)
    db.session.commit()
    return jsonify({'message': 'Task added successfully!'}), 201

# Видалити завдання
@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    task = Task.query.get(task_id)
    if task is None:
        return jsonify({'message': 'Task not found!'}), 404
    db.session.delete(task)
    db.session.commit()
    return jsonify({'message': 'Task deleted successfully!'}), 200

if __name__ == '__main__':
    app.run(debug=True)
