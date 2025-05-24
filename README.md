# Text-to-SQL Python Model

This project implements a Text-to-SQL model using Python, leveraging HuggingFace's `datasets` library and PyTorch for training and evaluation. The model combines datasets like Spider and WikiSQL to train a robust Text-to-SQL system.

## Features
- Combines multiple datasets (e.g., Spider and WikiSQL) for training.
- Uses HuggingFace's `datasets` library for dataset management.
- Supports PyTorch for model training.
- Configurable training arguments using HuggingFace's `Trainer`.

## Requirements
The project requires the following dependencies:
- Python 3.8+
- PyTorch
- HuggingFace `transformers` and `datasets`
- Other dependencies listed in `requirements.txt`

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/agbanil/TextToSqlPythonModel.git
   cd TextToSqlPythonModel
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the datasets:
   - Ensure the Spider and WikiSQL datasets are available.
   - Combine them using the HuggingFace `datasets.concatenate_datasets` function.

4. Train the model:
   - Configure training arguments in the script.
   - Run the training script.

## File Structure
- `script.py`: Main script for dataset preparation and model training.
- `requirements.txt`: Lists all required dependencies.

## Notes
- Ensure the combined dataset is converted to PyTorch tensors before training.
- Adjust `TrainingArguments` to match your hardware and training preferences.

## Troubleshooting
- If you encounter an "unexpected argument" error in `TrainingArguments`, ensure all arguments are valid and compatible.
- For dataset issues, verify that the datasets are properly formatted and combined.

## License
This project is licensed under the MIT License.
