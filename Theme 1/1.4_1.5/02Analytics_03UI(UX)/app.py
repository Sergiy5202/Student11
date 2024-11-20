from flask import Flask, jsonify, request, render_template
import psycopg2

app = Flask(__name__)

# Connect to the PostgreSQL database
def connect_db():
    return psycopg2.connect(
        database="mi2024",
        user="postgres",  # Ваше ім'я користувача PostgreSQL
        password="admin",  # Ваш пароль PostgreSQL
        host="localhost",
        port="5432"
    )

# Home route to render the UI
@app.route('/')
def home():
    return render_template('index.html')

# Get all documents
@app.route('/api/documents', methods=['GET'])
def get_documents():
    connection = connect_db()
    cursor = connection.cursor()
    query = '''
    SELECT d.document_id, d.number, t.type_name, d.content, d.date, d.completion_status
    FROM documents d
    JOIN document_types t ON d.type_id = t.type_id;
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = ['document_id', 'number', 'type_name', 'content', 'date', 'completion_status']
    documents = [dict(zip(columns, row)) for row in rows]
    cursor.close()
    connection.close()
    return jsonify(documents)

if __name__ == '__main__':
    app.run(debug=True)
