import random
import string
import tqdm
from tokenizer import TinypyTokenizer

tpt = TinypyTokenizer()

CONTEXT_WINDOW = 128
MIN_LENGTH = 1
MAX_LENGTH = 5
MIN_STRINGS = 2
MAX_STRINGS = 5
NUM_EXAMPLES = 1_000_000
OUTPUT_PATH = "./data.txt"
RANDOM_SEED = 97
BASE_DATA = "../d10-push-aquos/data.txt"


def random_string(min_length: int, max_length: int) -> str:
    length = random.randint(min_length, max_length)
    letters = random.choices(string.ascii_lowercase, k=length)
    return "".join(letters)


def build_example(
    min_length: int,
    max_length: int,
    min_strings: int,
    max_strings: int,
) -> str:
    count = random.randint(min_strings, max_strings)
    parts = [random_string(min_length, max_length) for _ in range(count)]
    input_block = "\n".join(parts)
    output_block = "\n".join(parts)
    return f"#input\n{input_block}\n#output\n{output_block}\n\n"


def generate_corpus(
    num_examples: int,
    min_length: int,
    max_length: int,
    min_strings: int,
    max_strings: int,
) -> str:
	with open(BASE_DATA, "r") as handle:
		data = handle.read()    
	s = set(data.split("\n\n")[:-1])
	examples = list()
	pbar = tqdm.tqdm(total=num_examples)
	while len(examples) < num_examples:
		example = build_example(min_length, max_length, min_strings, max_strings)
		if example in s:
			continue
		if len(tpt.tokenize(example)) > CONTEXT_WINDOW:
			continue
		pbar.update(1)
		s.add(example)
		examples.append(example)
	return "".join(examples)


def main() -> None:
    random.seed(RANDOM_SEED)
    corpus = generate_corpus(
        NUM_EXAMPLES,
        MIN_LENGTH,
        MAX_LENGTH,
        MIN_STRINGS,
        MAX_STRINGS,
    )
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(corpus)


if __name__ == "__main__":
    main()
