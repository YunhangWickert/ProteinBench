# ProteinBench

> **ProteinBench**: A comprehensive benchmark suite for protein-based models

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/YunhangWickert/ProteinBenchmark.git
cd ProteinBenchmark

# Install Python dependencies
conda env create -f environment.yaml
conda activate proteinbench
```
## 🚀 Quick Start
## Tasks at the functional level Evaluation
### Fine-tuning a single task

```bash
cd ProteinBench/tasks
env CUDA_VISIBLE_DEVICES=0  python main.py --eval_model function-eval --pretrain_model_name esm2_650m --lr 1e-4 --lr_scheduler cosine --config_name fitness_prediction
```

## Structure --> Sequence Evaluation

```bash
cd ProteinBench/tasks
env CUDA_VISIBLE_DEVICES=0  python main.py --eval_model structure-to-sequence-eval --pretrain_model_name AlphaDesign --lr 1e-3 --lr_scheduler onecycle
```
## Sequence --> Structure Evaluation
### Unconditional Generation

- Step 0: Remove old files of history evaluation (if exist): ```designs/,sequences/,scores/,aln.m8,info.csv,tm_info.csv,final_result_un.csv,/jsonl```
- Step 1: Place the predicted proteins to be evaluated in the folder ```evaluate/workspace/pdbs```.
- Step 2: Install [FoldSeek](https://github.com/steineggerlab/foldseek) and download pre-generated databases
```shell
  cd ProteinBench/evaluate/workspace/fs_db
  foldseek databases PDB pdb tmp
```
- Step 3: Run the evaluation shell script 
```shell
  cd evaluate/
  sh run_evaluation_un.sh
```
The evaluation results will be written into ```evaluate/workspace/final_result_un.csv``` 
