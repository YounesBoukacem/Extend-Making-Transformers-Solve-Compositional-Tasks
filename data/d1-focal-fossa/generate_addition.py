import random


MIN_DIGITS = 1
MAX_DIGITS = 5
MIN_NUMBERS = 2
MAX_NUMBERS = 4
NUM_EXAMPLES = 1000000
OUTPUT_PATH = "./data.txt"
RANDOM_SEED = 1234


def random_number(min_digits: int, max_digits: int) -> int:
    digits = random.randint(min_digits, max_digits)
    if digits == 1:
        return random.randint(0, 9)
    lower = 10 ** (digits - 1)
    upper = (10**digits) - 1
    return random.randint(lower, upper)


def build_example(min_digits: int, max_digits: int, min_numbers: int, max_numbers: int) -> str:
    count = random.randint(min_numbers, max_numbers)
    numbers = [random_number(min_digits, max_digits) for _ in range(count)]
    input_block = "\n".join(str(num) for num in numbers)
    output_value = sum(numbers)
    return f"#input\n{input_block}\n#output\n{output_value}"


def generate_corpus(
    num_examples: int,
    min_digits: int,
    max_digits: int,
    min_numbers: int,
    max_numbers: int,
) -> str:
    examples = [
        build_example(min_digits, max_digits, min_numbers, max_numbers)
        for _ in range(num_examples)
    ]
    return "\n\n".join(examples) + "\n\n"


def main() -> None:
    random.seed(RANDOM_SEED)
    corpus = generate_corpus(
        NUM_EXAMPLES,
        MIN_DIGITS,
        MAX_DIGITS,
        MIN_NUMBERS,
        MAX_NUMBERS,
    )
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(corpus)


if __name__ == "__main__":
    main()
