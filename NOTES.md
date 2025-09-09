## Notes

- Data consists of 3235 dialogs in restaurant domain.
- Each dialog is an interaction between user and system (exchange of utterances).  <br /> <br />

- In part 1, we will implement a ML classifier to classify dialog acts in dialog data.
- Examples of dialog acts: greeting, asking, confirming, etc.
- There are 15 acts (labels) total in the data
<br /> <br />

### Data format:  
dialog_act [space] utterance_content  
dialog_act [space] utterance_content  
... <br /> <br />

### We will split the data into:
1. Train set (75%)
2. Validation set for tuning (10%)
3. Test set (15%)

### Explicit instructions
- ALL Data needs to be converted to lowercase!
- ALL user input also needs to be converted to lowercase!
- If utterance is labeled with 2 different dialog acts --> Only use first dialog act!

### 2 Baseline systems
1. MAJORITY LABEL - regardless of the content of the utterance, always assigns the majority class in the data. In the current dataset this is the 'inform' label.
2. MANUAL RULE-BASED SYSTEM (based on keyword matching) - Example rule: anytime an utterance contains ‘goodbye’, it would be classified with dialog act (label) 'bye'. This baseline can be made iteratively, create an initial version, test the performance, and then add or remove keywords for specific classes to improve the results. A reasonable performance for this baseline is >= 0.80.

## Our code should:
Offer a prompt to enter a new utterance (via cmd line) and classify this utterance, and repeat the prompt until the user exits.


## Machine learning (we will train 3 classifiers)
Possible classifiers include (but are not limited to) Decision Trees, Logistic Regression, or a Feed Forward neural network.  

We need to use a bag of words representation as input for a classifier. Depending on the classifier and setup of your ML pipeline you may need to keep an integer (for example 0) for out-of-vocab words (e.g when a test sentence is entered that contains a word which was not in the training data), assign the special integer.  

After training, testing, and reporting performance, the program should offer a prompt to enter a new sentence and classify this sentence, and repeat the prompt until the user exits.  

- Many utterances in the dialogs are not unique (same sentence is spoken by users in different dialogs). This influences ML, because even with a train-test split the same sentence may appear in both the train and test set.

- Create a second dataset with duplicates removed and create a second train-test split after removing the duplicates  

- Build and evaluate 2 different variants of each model, one with the original data and split, one with the deduplicated data and split.  

- Discuss the differences in performance between each pair of variant models in the report.