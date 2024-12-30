from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-vi'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Prepare data and fine-tune


# push to hub
model.push_to_hub("datngo2001/opus-mt-en-vi-finetuned")
tokenizer.push_to_hub("datngo2001/opus-mt-en-vi-finetuned")
