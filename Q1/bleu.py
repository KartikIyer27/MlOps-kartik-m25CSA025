import sacrebleu

def read_lines(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    predictions = read_lines("output.txt")
    references = read_lines("reference.txt")

    bleu = sacrebleu.corpus_bleu(predictions, [references])

    print("BLEU Score:", bleu.score)

if __name__ == "__main__":
    main()