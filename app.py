from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import torch
import re

# Initialize app
app = FastAPI(title="Text Summarization System", description="Summarize dialogues with T5!", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & tokenizer
model_path = "./saved_summary_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Templates directory
templates = Jinja2Templates(directory="templates")


# Pydantic input schema
class DialogueInput(BaseModel):
    dialogue: str


# Clean text utility
# def clean_text(text: str) -> str:
#     text = re.sub(r'\r\n', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'<.*?>', '', text)
#     return text.strip().lower()


def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip().lower()


# Summarization function
def summarize_dialogue(dialogue: str) -> str:
    dialogue = "summarize: " + clean_text(dialogue)
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


# Summarize API endpoint
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


# Web UI endpoint
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
