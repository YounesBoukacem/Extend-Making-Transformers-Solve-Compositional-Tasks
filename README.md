# Extend-Making-Transformers-Solve-Compositional-Tasks
This repository contains the code and presentation content (report, poster, slides, video) for our validation project of the course "Introduction to deep learning" at Master 2 MVA 2025/2026 taught by Vincent Lepetit and Maria Vakalopoulou.

# The code part
The code part of the repository is structured around two folders:

- 📁 `data/`: contains all the generated datasets (for training and testing).

- 📁 `experiments/archive`: contains all the experiments that were conducted, both "draft" experiments (to be ignored) and "final" experiments which results are delivered in the final report, and which are:
    - 📁 `a2-push-nova`, `a3-reaper-abyss`, and `a4-ribon-moon`: these 3 archives contain the experiments for training with next token prediction for the APE, RPE and RPB models respectively.
    - 📁 `a6-sopush-nova`, `a7-soreaper-abyss`, and `a8-soribon-moon`: these 3 archives contain the experiments for additional training with auto-regressive input masking for the APE, RPE and RPB models respectively.

Each experiment archive contains 4 `experiments` subfolders:

- 📁 `e-mechtogan`: contain the code for training and testing in the case of the concatenation task. A dedicated subfolder contains the evaluation for the In-Distribution test set, and another dedicated subfolder contains the evaluations for all the Out-of-distribution data benchmarks.

- 📁 `e-pegasus`: contain the code for training and testing in the case of the interleaving task. A dedicated subfolder contains the evaluation for the In-Distribution test set, and another dedicated subfolder contains the evaluations for all the Out-of-distribution data benchmarks.

- 📁 `e-ventus`: contain the code for training and testing in the case of the reversing task. A dedicated subfolder contains the evaluation for the In-Distribution test set, and another dedicated subfolder contains the evaluations for all the Out-of-distribution data benchmarks.

- 📁 `e-equinox`: contain the code for training and testing in the case of the duplication task. A dedicated subfolder contains the evaluation for the In-Distribution test set, and another dedicated subfolder contains the evaluations for all the Out-of-distribution data benchmarks.

**N.B:** the names of the data, archive and experiments folders were randomly generated unique IDs (this is why they don't really make sense 😅).

**N.B:** the model checkpoints and datasets weren't loaded to this repository as they are too heavy, but we ensured that they are completely reproducible by fixing the random seeds in the script 😄.

# The presentation part

The 📁 `presentation` contains the required:
- ⏯️ `Video.mp4` video presentation
- 🎬 `Slides.pdf` used in the video
- 📄 `Report.pdf`
- 📃 `Poster.pdf`