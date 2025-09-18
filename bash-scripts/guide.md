### Bash-version of Training/Inference

Pre-requisites:
* Conda
```bash
conda create -n nmt python=3.6 fairseq pytorch sacrebleu -c pytorch -c nvidia -c conda-forge
```

Use `sentencepiece`:
```bash
conda activate nmt
conda install -c conda-forge sentencepiece
```

Download the pre-trained model:
```bash
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz tar -xzvf mbart.CC25.tar.gz
```

Note: Take note of the directory. Ideally, you should store it in `mbart` folder, in the same level as `bash-scripts`. **NOT INSIDE**!

`mbart` folder should contain 3 files:
- `model.pt`
- `sentence.bpe.model`
- `dict.txt`

Create folders inside `mbart`:
- `bpe`
- `bin`
- `checkpoints`

Refer to the bash scripts.

# Run in order:
Before running: 
```bash
chmod +x <file>.sh
```

To run: 
```bash
./<file>.sh
```
1. `preprocess.sh`
2. `binarize.sh`
3. `train.sh`
4. `translate.sh`
5. `prep-inference.sh`