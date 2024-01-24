from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-large")

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

new_tokens = ["[W_PED]", "[W_TR]", "[POL]"]

tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
model.resize_token_embeddings(len(tokenizer))


tokenizer.save_pretrained("./klue_roberta_token")
model.save_pretrained("./klue_roberta_token")
