

\\\\ How to Run the Project

1. 📦 Install Dependencies

Make sure Python 3.8+ is installed, then install the required packages:
pip install -r requirements.txt
 2. 🧠 Start the Backend Server (FastAPI)

Run the backend API using:

uvicorn backend:app --reload

This starts the server at:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

 3. 🌐 Launch the Frontend

Open the `index.html` file directly in your browser (just double-click it or right-click → "Open with browser").

> This HTML page lets you enter a paragraph and a question, and it will display the answer after calling the backend API.
 4. 🧪 (Optional) Run via Command Line

If you want to test the model using the terminal:

bash
python predict.py and main.py

You’ll be prompted to enter a context and a question. The model will return the answer in the terminal
