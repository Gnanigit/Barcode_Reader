document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('barcode-image');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('http://127.0.0.1:5000/scan-barcode', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        document.getElementById('result').textContent = JSON.stringify(data, null, 2);
    } catch (err) {
        document.getElementById('result').textContent = 'Error: ' + err.message;
    }
});
