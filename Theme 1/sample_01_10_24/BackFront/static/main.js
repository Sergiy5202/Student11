document.addEventListener('DOMContentLoaded', () => {
    fetchSensors();
    document.getElementById('addSensorForm').addEventListener('submit', addSensor);
});

async function fetchSensors() {
    const response = await fetch('/api/sensors');
    const sensors = await response.json();
    const tableBody = document.getElementById('sensorsTableBody');
    tableBody.innerHTML = '';
    sensors.forEach(sensor => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${sensor.sensor_id}</td>
            <td>${sensor.location}</td>
            <td>${sensor.type}</td>
            <td>${sensor.status ? 'Active' : 'Inactive'}</td>
            <td><button onclick="deleteSensor(${sensor.sensor_id})">Delete</button></td>
        `;
        tableBody.appendChild(row);
    });
}

async function addSensor(event) {
    event.preventDefault();
    const location = document.getElementById('location').value;
    const type = document.getElementById('type').value;
    const status = document.getElementById('status').checked;
    const response = await fetch('/api/sensors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ location, type, status })
    });
    if (response.ok) {
        fetchSensors();
    }
}

async function deleteSensor(sensor_id) {
    const response = await fetch(`/api/sensors/${sensor_id}`, {
        method: 'DELETE'
    });
    if (response.ok) {
        fetchSensors();
    }
}
