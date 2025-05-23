# 📝 Real-Time Text Summarization App

This is a real-time dialogue summarization web app built using:

- 🔥 FastAPI (backend API)
- 🎯 Hugging Face Transformers (T5 model for summarization)
- 🎨 HTML/CSS + JS (frontend interface)
- 🚀 Trained on the [Samsum Dialogue Dataset](https://www.kaggle.com/datasets/cpllab/samsum-dataset)

---

## 🔧 Features

- Input any human dialogue, chat, or conversation.
- Summarizes it using a pre-trained or custom fine-tuned T5 model.
- FastAPI + Jinja2 templated HTML frontend.
- Easily deployable as an API or full web app.

---

## 📁 Project Structure

text_summarization_app/
├── main.py # FastAPI backend
├── templates/
│ └── index.html # Frontend HTML
├── saved_summary_model/ # Fine-tuned T5 model directory
├── requirements.txt
└── README.md

## 🧪 How to Run Locally

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/text-summarization-app.git
cd text-summarization-app

pip install -r requirements.txt

uvicorn main:app --reload

Open in Browser
Go to: http://127.0.0.1:8000

