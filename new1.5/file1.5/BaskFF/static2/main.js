// Отримати всі дані з датчиків
async function getSensorData() {
    const response = await fetch('/api/sensors');
    const data = await response.json();
    const dataDiv = document.getElementById('sensor-data');
    dataDiv.innerHTML = data.map(d => 
        `<p>ID: ${d.data_id}, Sensor ID: ${d.sensor_id}, Temp: ${d.temperature}, Humidity: ${d.humidity}, Timestamp: ${d.timestamp} 
        <button onclick="deleteSensorData(${d.data_id})">Видалити</button></p>`
    ).join('');
}

// Додати новий запис з датчика
document.getElementById('add-sensor-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const sensor_id = document.getElementById('sensor_id').value;
    const temperature = document.getElementById('temperature').value;
    const humidity = document.getElementById('humidity').value;

    await fetch('/api/sensors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sensor_id, temperature, humidity })
    });
    getSensorData();
});

// Видалити запис
async function deleteSensorData(data_id) {
    await fetch(`/api/sensors/${data_id}`, { method: 'DELETE' });
    getSensorData();
}

// Отримати дані під час завантаження сторінки
getSensorData();
