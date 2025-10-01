from transformers import pipeline

generator = pipeline("text-generation", model = "gpt2")

prompt = input("Input the text you want to make story on:")
len = int(input("Input the length of the story you want to generate:"))

result = generator(prompt, max_length = len, num_return_sequences = 2,max_new_tokens=700,
    truncation=True,
    pad_token_id=50256,
    temperature=0.9,   # higher = more creative
    top_p=0.95)


print(result[0]['generated_text']) 
