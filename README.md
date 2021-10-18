DeepCoNN
===
The code implementation for the paper：  
Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation. In WSDM. ACM, 425-434.

# Environments
  + python 3.8
  + pytorch 1.70

# Dataset
  You need to prepare the following documents:  
  1. dataset(`data/Digital_Music_5.json.gz`)  
   Download from http://jmcauley.ucsd.edu/data/amazon (Choose Digital Music)  
   Preprocess origin dataset in json format to train.csv,valid.csv and test.csv.  
   ```
   python preprocess.py
   ```

  2. Word Embedding(`embedding/glove.6B.50d.txt`)  
   Download from https://nlp.stanford.edu/projects/glove

# Running

Train and evaluate the model:
```
python main.py
```
