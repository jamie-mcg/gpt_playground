import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_text(prompt, model_name="gpt2", max_length=50, num_return_sequences=1):
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text using sampling
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=True,  # Enable sampling
            top_k=50,  # Top-k sampling
            top_p=0.95,  # Top-p (nucleus) sampling
        )

    # Decode the generated text
    generated_texts = [
        tokenizer.decode(output[i], skip_special_tokens=True)
        for i in range(num_return_sequences)
    ]
    return generated_texts


if __name__ == "__main__":
    prompt = "Once upon a time"
    generated_texts = generate_text(prompt, max_length=100, num_return_sequences=3)

    for i, text in enumerate(generated_texts):
        print(f"Generated Text {i+1}:\n{text}\n")
