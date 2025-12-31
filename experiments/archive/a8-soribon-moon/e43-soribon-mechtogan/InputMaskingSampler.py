from tokenizer import TinypyTokenizer
import tqdm
import numpy as np

class InputMaskingSampler:

	def __init__(self, data_path, batch_size, block_size, nb_examples=None):
		
		print('[*]Initializing InputMaskingSampler ...')
		self.tpt = TinypyTokenizer()
		self.data_path = data_path
		self.batch_size = batch_size
		self.block_size = block_size

		print('[*]Reading data ...')
		with open(data_path, 'r') as f:
			data = f.read()
		
		print('[*]Preparing examples ...')
		examples = data.split('\n\n')[:-1][:nb_examples]
		examples = np.random.permutation(examples)
		encoded_examples = []
		self.EOI_tokens_indices = []
		for example in tqdm.tqdm(examples):
			example = example + "\n\n"
			tokenized_example = self.tpt.tokenize(example)
			tokenized_example = tokenized_example + self.tpt.tokenize("\n\n" * (self.block_size - len(tokenized_example)))
			encoded_examples.append(self.tpt.encode_tokens_list(tokenized_example))
			self.EOI_tokens_indices.append(tokenized_example.index('#output\n'))
		self.encoded_examples = np.array(encoded_examples, dtype=np.uint8)

	
	def __iter__(self):
		return self


	def __next__(self):
		# Randomly select batch_size examples
		selected_examples = np.random.choice(self.encoded_examples.shape[0], size=self.batch_size, replace=False)
		x = self.encoded_examples[selected_examples]
		last_tokens_next_token = np.full((self.batch_size, 1), self.tpt.encod_map['\n\n'], dtype=np.uint8)
		y = np.concatenate((x[:, 1:], last_tokens_next_token), axis=1)
		return x, y, [self.EOI_tokens_indices[selected_example] for selected_example in selected_examples]