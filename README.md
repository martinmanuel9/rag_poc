# Test Bench for RAG Model 
## Setting Up the Environment
1. Utilize Anaconda 3 
2. Install Python v11 
3. Install Pytorch 
```python
conda install pytorch torchvision -c pytorch
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
```
4. We utilize the original HuggingFace RAG model 
5. Using the following Kaggle Dataset to test/train etc: 
https://www.kaggle.com/competitions/kaggle-llm-science-exam/data?select=test.csv

# TODO
## Classification
1. Are the generated answers correct 
   1. Need to add performance metrics on there AUC/ROC scores, F1 Scores, Matthews_CorrCoef
2. Need to test the data 

## Document Pipeline
1. Determine how to add new questions or new document into corpus 
2. Figure out how to add to this
3. New criteria for answering questions? 