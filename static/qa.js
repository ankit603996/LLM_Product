document.addEventListener('DOMContentLoaded', () => {
    const questionForm = document.getElementById('question-form');
    const submitButton = document.getElementById('submit-button');
    const questionProgress = document.getElementById('question-progress');
    const ragAnswer = document.getElementById('rag-answer');
    const directAnswer = document.getElementById('direct-answer');

    questionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = document.getElementById('question-input').value;

        ragAnswer.innerText = '';
        directAnswer.innerText = '';
        questionProgress.style.display = 'block';
        submitButton.disabled = true;

        try {
            const [ragResponse, directResponse] = await Promise.all([
                fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                }),
                fetch('/ask_direct', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                })
            ]);

            const ragResult = await ragResponse.json();
            const directResult = await directResponse.json();

            ragAnswer.innerText = ragResult.answer || ragResult.error;
            directAnswer.innerText = directResult.answer || directResult.error;
        } catch (error) {
            ragAnswer.innerText = 'Error getting response';
            directAnswer.innerText = 'Error getting response';
        } finally {
            questionProgress.style.display = 'none';
            submitButton.disabled = false;
        }
    });
});