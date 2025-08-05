# main.py
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: Load the SQuAD dataset
print("📦 Loading SQuAD dataset...")
dataset = load_dataset("squad")

# Step 2: Load the tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Step 3: Preprocessing function
def preprocess(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    answers = examples["answers"]

    inputs = tokenizer(
        questions,
        contexts,
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end token index in context
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(offsets) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            for idx in range(token_start_index, token_end_index + 1):
                if offsets[idx][0] <= start_char and offsets[idx][1] > start_char:
                    start_pos = idx
                if offsets[idx][0] < end_char and offsets[idx][1] >= end_char:
                    end_pos = idx
                    break
            start_positions.append(start_pos)
            end_positions.append(end_pos)

    inputs["start_positions"] = torch.tensor(start_positions)
    inputs["end_positions"] = torch.tensor(end_positions)
    return inputs

# Step 4: Tokenize and preprocess dataset
print("🔄 Tokenizing dataset...")
tokenized_train = dataset["train"].select(range(1000)).map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
tokenized_val = dataset["validation"].select(range(100)).map(preprocess, batched=True, remove_columns=dataset["validation"].column_names)

# Step 5: Training configuration
training_args = TrainingArguments(
    output_dir="./qa_model",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_total_limit=1,
    logging_dir='./logs',
    logging_steps=10
)

# Step 6: Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val
)

# Step 7: Train and save model
print("🚀 Training model...")
trainer.train()

print("💾 Saving model to './qa_model/' ...")
model.save_pretrained("qa_model")
tokenizer.save_pretrained("qa_model")
