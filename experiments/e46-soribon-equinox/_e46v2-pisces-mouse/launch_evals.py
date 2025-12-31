# Create a subfolder for each eval i.e. for each test set
# Launch the eval script in each subfolder
import os
import shutil
from pathlib import Path

TESTS_ROOT_FOLDER = "/data/yb2618/DL_project_MVA/data/d14-aquos-bench/"
DDIR = "/data/yb2618/DL_project_MVA/data/d10-push-aquos/"

test_sets = [
	"ood_nb_inputs.txt",
	"ood_str_length_only_one.txt",
	"ood_str_length_all.txt",
	"ood_nb_inputs_str_length_only_one.txt",
	"ood_nb_inputs_str_length_all.txt"	
]

deviceid = 2
for i, test_set in enumerate(test_sets):
	
	# Create the eval subfolder
	eval_name = test_set.replace(".txt", "")
	eval_folder = Path(f"./{i+1}_{eval_name}")
	eval_folder.mkdir(parents=True, exist_ok=True)
	
	# Copy the eval script into the eval folder
	shutil.copyfile("./eval.py", eval_folder / "eval.py")
	shutil.copyfile("./tokenizer.py", eval_folder / "tokenizer.py")
	
	# Launch the script in the eval folder
	# Make sure the python instance is launched from within the eval folder (i.e. the cwd is the eval folder)
	print("\n==========================================================")
	print(f"Launching eval for test set: {test_set}")
	print("==========================================================\n")
	os.system(f"cd {eval_folder} && python eval.py --test_set_path {TESTS_ROOT_FOLDER}{test_set} --ddir {DDIR} --deviceid {deviceid}")