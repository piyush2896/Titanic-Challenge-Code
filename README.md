# Kaggle Titanic Challenge
This repository contains code to tackle the [Titanic Challenge](https://www.kaggle.com/c/titanic).<br/>

## Data Usage
The Code removes Name and Ticket Column as of now. Also, it converts **male** value to 1 and **female** to 2. Why 1 or 2? You can say, I was bore with 1 and 0.
Same goes for Embarkments; **C** to 1, **Q** to 2 and **S** to 3. The code uses Gaussian Naive Bayes from sklearn library.

## Usage
Run main.py to generate result.csv.

## Current Accuracy
The current accuracy of the code is **0.74641**. Still trying to improve it.