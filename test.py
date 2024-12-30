from transformers import MarianMTModel, MarianTokenizer

model = MarianMTModel.from_pretrained("datngo2001/opus-mt-en-vi-finetuned")
tokenizer = MarianTokenizer.from_pretrained("datngo2001/opus-mt-en-vi-finetuned")

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt", padding=True)
outputs = model.generate(**inputs)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated_text)