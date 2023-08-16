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
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, StoppingCriteriaList, MaxLengthCriteria

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

input_text = "Who is the inventor of the light bulb?"

# Tokenize input using the tokenizer
inputs = tokenizer(input_text, return_tensors="pt")

# Set up stopping criteria (as a list)
stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=100)])

# Generate with stopping criteria
generated = model.generate(input_ids=inputs.input_ids, stopping_criteria=stopping_criteria)

# Decode generated output
decoded_output = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print(decoded_output)
