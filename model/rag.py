"""
Application:        Retrieval-Augmented Generation (RAG) Proof-of-Concept (POC)
File name:          rag.py
Author:             Martin Manuel Lopez
Creation:           08/12/2023

"""
# MIT License
#
# Copyright (c) 2023
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

####################################################################################################
import pandas as pd
import torch
from transformers import RagTokenizer, RagTokenForGeneration
from transformers import RagRetriever, RagTokenizer, RagTokenForGeneration
from datasets import load_dataset

# Initialize tokenizer, retriever, and model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
model.set_retriever(retriever)


train_csv_file_path = "./documents/train.csv"  # Replace with the actual path
train_df = pd.read_csv(train_csv_file_path)

# Prepare data for training
questions = train_df["prompt"].tolist()
potentialAnswers = train_df[["A", "B", "C", "D"]].values.tolist()
correctAnswers = train_df["answer"].tolist()

# Encode the data
encoded_prompts = tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
encoded_answers = tokenizer(correctAnswers, padding=True, truncation=True, return_tensors="pt")

# Going through the exam and answering those questions
for i in range(len(questions)):
    input_ids = encoded_prompts["input_ids"][i]
    # Generate text using the RAG model
    generated = model.generate(input_ids.unsqueeze(0))  # Adjust max_length as needed
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("Question:\n", questions[i], "\n", "Generated Text:", generated_text)