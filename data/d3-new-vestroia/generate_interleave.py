import random
import string


MIN_LENGTH = 2
MAX_LENGTH = 5
MIN_STRINGS = 2
MAX_STRINGS = 4
NUM_EXAMPLES = 1_000_000
OUTPUT_PATH = "./data.txt"
RANDOM_SEED = 1234


def random_string(length: int) -> str:
    letters = random.choices(string.ascii_lowercase, k=length)
    return "".join(letters)


def interleave(parts: list[str]) -> str:
    return "".join("".join(group) for group in zip(*parts))


def build_example(min_length: int, max_length: int, min_strings: int, max_strings: int) -> str:
    length = random.randint(min_length, max_length)
    count = random.randint(min_strings, max_strings)
    parts = [random_string(length) for _ in range(count)]
    input_block = "\n".join(parts)
    output_value = interleave(parts)
    return f"#input\n{input_block}\n#output\n{output_value}"


def generate_corpus(
    num_examples: int,
    min_length: int,
    max_length: int,
    min_strings: int,
    max_strings: int,
) -> str:
    examples = [
        build_example(min_length, max_length, min_strings, max_strings)
        for _ in range(num_examples)
    ]
    return "\n\n".join(examples) + "\n\n"


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
