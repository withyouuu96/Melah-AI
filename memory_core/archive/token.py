from transformers import AutoTokenizer

# ลองใช้ Qwen หรือ GPT-3/4 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat") # หรือ 'gpt2'
with open('legacy_91_session.txt', 'r', encoding='utf-8') as f:
    text = f.read()
tokens = tokenizer.encode(text)
print("Token count:", len(tokens))
