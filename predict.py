# predict.py

from transformers import pipeline

print("\n🧠 Question Answering System (Pretrained)")
print("------------------------------------------")

# Load pipeline with pretrained model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Take user inputs
context = input("📜 Enter the paragraph (context):\n")
question = input("❓ Enter your question:\n")

# Run inference
result = qa_pipeline(question=question, context=context)

# Show result
print("\n✅ Answer:", result["answer"])
