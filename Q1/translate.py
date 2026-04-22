from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "Helsinki-NLP/opus-mt-bn-en"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

def translate_line(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs, max_length=128)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def main():
    tokenizer, model = load_model()

    with open("input.txt", "r", encoding="utf-8") as f:
        bengali_lines = [line.strip() for line in f if line.strip()]

    translated_lines = []
    for line in bengali_lines:
        translated = translate_line(line, tokenizer, model)
        translated_lines.append(translated)

    with open("output.txt", "w", encoding="utf-8") as f:
        for line in translated_lines:
            f.write(line + "\n")

    print("done Output saved to output.txt")

    if translated_lines:
        print("\nFirst statement:")
        print(translated_lines[0])

if __name__ == "__main__":
    main()