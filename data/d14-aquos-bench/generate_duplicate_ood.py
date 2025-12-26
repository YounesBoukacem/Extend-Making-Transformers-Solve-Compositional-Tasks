import random
import string
import tqdm
from tokenizer import TinypyTokenizer

tpt = TinypyTokenizer()

CONTEXT_WINDOW = 128
RANDOM_SEED = 1234


def random_string(min_length: int, max_length: int) -> str:
    length = random.randint(min_length, max_length)
    letters = random.choices(string.ascii_lowercase, k=length)
    return "".join(letters)


def build_ood_example(
	ood_length_mode: str, # can either be "all" or "only-one"
	old_min_length: int,
	old_max_length: int,
	min_length: int,
	max_length: int,
	min_strings: int,
	max_strings: int,
) -> str:
	count = random.randint(min_strings, max_strings)
	if ood_length_mode == "all":
		parts = [random_string(min_length, max_length) for _ in range(count)]
	else:  # only one
		parts = [random_string(old_min_length, old_max_length) for _ in range(count)]
		parts[random.randint(0, count - 1)] = random_string(min_length, max_length)	
	input_block = "\n".join(parts)
	output_value = "\n".join(parts)
	return f"#input\n{input_block}\n#output\n{output_value}\n\n"

def generate_corpus(
	ood_length_mode: str,
	old_min_length: int,
	old_max_length: int,
	num_examples: int,
	min_length: int,
	max_length: int,
	min_strings: int,
	max_strings: int,
) -> str:
	examples = list()
	s = set()
	pbar = tqdm.tqdm(total=num_examples)
	while len(examples) < num_examples:
		example = build_ood_example(ood_length_mode, old_min_length, old_max_length, min_length, max_length, min_strings, max_strings)
		if example in s:
			continue
		if len(tpt.tokenize(example)) > CONTEXT_WINDOW:
			continue
		pbar.update(1)
		s.add(example)
		examples.append(example)
	return "".join(examples)

configs = [
	# ood_nb_inputs
	{	
		"OOD_LENGTH_MODE": "only-one",
		"OLD_MIN_LENGTH": 1,
		"OLD_MAX_LENGTH": 5,
		"NUM_EXAMPLES": 10_000,
  		"MIN_LENGTH": 1,
  		"MAX_LENGTH": 5,
  		"MIN_STRINGS": 6,
  		"MAX_STRINGS": 10,
  		"OUTPUT_PATH": "./ood_nb_inputs.txt"
	},

	# ood_str_length_only_one
	{	
		"OOD_LENGTH_MODE": "only-one",
		"OLD_MIN_LENGTH": 1,
		"OLD_MAX_LENGTH": 5,
		"NUM_EXAMPLES": 10_000,
  		"MIN_LENGTH": 6,
  		"MAX_LENGTH": 10,
  		"MIN_STRINGS": 2,
  		"MAX_STRINGS": 5,
  		"OUTPUT_PATH": "./ood_str_length_only_one.txt"
	},

	# ood_str_length_all
	{
		"OOD_LENGTH_MODE": "all",
		"OLD_MIN_LENGTH": 1,
		"OLD_MAX_LENGTH": 5,
		"NUM_EXAMPLES": 10_000,
  		"MIN_LENGTH": 6,
  		"MAX_LENGTH": 10,
  		"MIN_STRINGS": 2,
  		"MAX_STRINGS": 5,
  		"OUTPUT_PATH": "./ood_str_length_all.txt"
	},

	# ood_nb_inputs_str_length_only_one
	{
		"OOD_LENGTH_MODE": "only-one",
		"OLD_MIN_LENGTH": 1,
		"OLD_MAX_LENGTH": 5,
		"NUM_EXAMPLES": 10_000,
  		"MIN_LENGTH": 6,
  		"MAX_LENGTH": 10,
  		"MIN_STRINGS": 6,
  		"MAX_STRINGS": 10,
  		"OUTPUT_PATH": "./ood_nb_inputs_str_length_only_one.txt"
	},

	# ood_nb_inputs_str_length_all
	{
		"OOD_LENGTH_MODE": "all",
		"OLD_MIN_LENGTH": 1,
		"OLD_MAX_LENGTH": 5,
		"NUM_EXAMPLES": 10_000,
  		"MIN_LENGTH": 6,
  		"MAX_LENGTH": 10,
  		"MIN_STRINGS": 6,
  		"MAX_STRINGS": 10,
  		"OUTPUT_PATH": "./ood_nb_inputs_str_length_all.txt"
	}

]

def main() -> None:
	random.seed(RANDOM_SEED)
	for config in configs:
		ood_length_mode = config["OOD_LENGTH_MODE"]
		old_min_length = config["OLD_MIN_LENGTH"]
		old_max_length = config["OLD_MAX_LENGTH"]
		num_examples = config["NUM_EXAMPLES"]
		min_length = config["MIN_LENGTH"]
		max_length = config["MAX_LENGTH"]
		min_strings = config["MIN_STRINGS"]
		max_strings = config["MAX_STRINGS"]
		output_path = config["OUTPUT_PATH"]

		corpus = generate_corpus(
			ood_length_mode,
			old_min_length,
			old_max_length,
			num_examples,
			min_length,
			max_length,
			min_strings,
			max_strings,
		)
		with open(output_path, "w", encoding="utf-8") as handle:
			handle.write(corpus)

if __name__ == "__main__":
	main()