Python 3.11.5 (v3.11.5:cce6ba91b3, Aug 24 2023, 10:50:31) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> import gradio as gr
... from transformers import RobertaTokenizer, RobertaForSequenceClassification
... import torch
... 
... # Load your trained model
... model = RobertaForSequenceClassification.from_pretrained("/kaggle/working/roberta-fake-news-model")
... tokenizer = RobertaTokenizer.from_pretrained("/kaggle/working/roberta-fake-news-model")
... 
... device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
... model.to(device)
... model.eval()
... 
... # Classification function
... def classify_news(title, content):
...     full_text = title + " " + content
...     inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
...     inputs = {k: v.to(device) for k, v in inputs.items()}
... 
...     with torch.no_grad():
...         logits = model(**inputs).logits
...         probs = torch.nn.functional.softmax(logits, dim=1)[0]
...         pred = torch.argmax(probs).item()
...         confidence = probs[pred].item() * 100
... 
...     label = "REAL" if pred == 1 else "FAKE"
... 
...     # 🔒 Add confidence warning logic
...     if confidence < 75:
...         return f"⚠️ Uncertain prediction: {label} — Confidence: {confidence:.2f}%\nPlease double-check this with a reliable source."
...     
SyntaxError: multiple statements found while compiling a single statement
>>>  return f"{label} — Confidence: {confidence:.2f}%"
... 

# Launch Gradio
gr.Interface(
    fn=classify_news,
    inputs=[
        gr.Textbox(label="News Title", placeholder="Enter the headline..."),
        gr.Textbox(lines=10, label="News Content", placeholder="Paste the full article here...")
    ],
    outputs="text",
    title="📰 Fake News Detector with RoBERTa",
    description="Enter the headline and article to check if it's REAL or FAKE."
).launch(share=True, inline=True)
SyntaxError: unexpected indent
