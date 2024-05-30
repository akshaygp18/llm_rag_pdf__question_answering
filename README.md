# LLM RAG PDF QUESTION ANSWERING MODEL
Question answering project


### STEP 01- Create a environment and activate the environment

```
conda create --prefix ./env python=3.9 -y
```

```
conda activate ./env
```
OR
```
source activate ./env
```

### STEP 02- install the requirements
```
pip install -r requirements.txt
```

### STEP 03- run the file 
```
uvicorn src.app:app --reload
```

