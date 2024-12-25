document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const fileNames = document.getElementById('file-names');
    const uploadResponse = document.getElementById('upload-response');
    const uploadProgress = document.getElementById('upload-progress');

    fileInput.addEventListener('change', () => {
        const files = Array.from(fileInput.files);
        fileNames.innerHTML = files.map(file => `<span>${file.name}</span>`).join('');
    });

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(uploadForm);

        uploadResponse.innerText = '';
        uploadResponse.className = '';
        uploadProgress.style.display = 'block';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            if (response.ok) {
                uploadResponse.className = 'success';
                uploadResponse.innerText = result.message;
                // Redirect to Q&A page after successful upload
                window.location.href = '/qa';
            } else {
                uploadResponse.className = 'error';
                uploadResponse.innerText = result.error;
            }
        } catch (error) {
            uploadResponse.className = 'error';
            uploadResponse.innerText = 'Error processing inputs';
        } finally {
            uploadProgress.style.display = 'none';
        }
    });
});
