Hybrid Semantic Retrieval & Intelligence System (HSRIS)
This project was built as part of Assignment 3 for Data Science for Software Engineering. The goal was to build a smart search system for customer support tickets that understands both keywords and meaning.

Student Info
Batch: 23F
Roll Number: 3001
What This Project Does
When a customer sends a support ticket, this system instantly finds the most similar past tickets so support agents can resolve issues faster.

It combines two powerful techniques:

TF-IDF finds tickets with matching keywords
GloVe understands the meaning behind words
So if someone types "money problem" the system also finds tickets about "billing issue" even though the exact words are different.

Dataset
Name: Customer Support Ticket Dataset
Source: Kaggle (by Suraj520)
Size: 8,469 records
Fields used: Ticket Description, Ticket Subject, Ticket Priority, Ticket Type, Ticket Channel
What I Built
Part 1: Categorical Encoding
Label Encoding for Ticket Priority Low=0, Medium=1, High=2, Critical=3
One-Hot Encoding for Ticket Channel Email=[0,1,0,0], Chat=[1,0,0,0] etc
Part 2: TF-IDF From Scratch
Custom regex tokenizer
Vocabulary of top 5000 words
Bigram and Trigram generator
TF-IDF matrix computed manually using NumPy
Stored as sparse tensor using torch.sparse
Part 3: GloVe Embeddings
Loaded GloVe 300d pretrained vectors
4142 out of 5000 words found in GloVe
858 OOV words handled with random vectors
TF-IDF weighted sentence embeddings
Hybrid Search
Final Score = 0.4 x TF-IDF + 0.6 x GloVe

GPU Optimization
torch.nn.DataParallel enabled on Dual T4 GPUs
100 queries processed in 0.141 seconds
Average 1.41ms per query
Results
Dataset: 8,469 tickets processed
Vocabulary: 5,000 words
Precision@5: 21.10%
Batch processing: 100 queries in 0.141 seconds
Platform: Kaggle Dual T4 GPU
GloVe vs TF-IDF Comparison
Query	TF-IDF Score	GloVe Score
money problem	0.4689	0.8811
device broken	0.2905	0.8723
cannot login	0.3796	0.9444
internet disconnecting	0.4970	0.8124
want money back	0.3687	0.8954
GloVe consistently scores 2x higher than TF-IDF!

Tech Stack
Python 3.12
PyTorch
NumPy
GloVe 840B 300d
Gradio 5.50.0
Kaggle Dual T4 GPU
Live App
Try it here: https://bc27d96384dd824269.gradio.live https://huggingface.co/spaces/Urba233017/HSRIS-Ticket-Search

How to Run
Open notebook on Kaggle
Add Customer Support Ticket Dataset
Enable GPU T4 x2 in settings
Run all cells from top to bottom
Wait for GloVe download (1GB)
Gradio app launches automatically
