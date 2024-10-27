# level 2. Open-Domain Question Answering (RAG)



## Train 

### MRC
```bash
python main.py mode=train
```

### DPR (Dense Passage Retrievel)
```bash
# bi-encoder dpr
python module/dpr/dense_ret.py 
# Colbert 
python module/colbert/train.py
```

## inference
```bash
python main.py mode=inference
```