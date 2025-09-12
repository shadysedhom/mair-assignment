# its unclear from the assignment if the most common label must be hardcoded or not, so I left it hardcoded
# only predict only the most common label and thus ignoring the utterance
def majority_baseline_predict(utterance, majority_label="inform"):
    return majority_label

# it was supprising how much address and phone number were used to request intent and thus impacted the accuracy greatly
def rule_based_predict(utterance, majority_label="inform"):
    utterance_words = utterance.lower().split()
    rules = {
        "hello": ["hello", "welcome", "hi"],
        "thankyou": ["thank", "thanks", "noice",],
        "affirm": ["yes", "correct", "yea", "ye"],
        "deny": ["no","not","dont want", "wrong","something else"],
        "bye": ["bye", "goodbye", "good bye"],
        "repeat": ["again", "repeat", "say that again", "once more"],
        "reqmore": ["more"],
        "reqalts": ["different","there another" ,"other", "alternatives", "alternative","anything else","what about","how about","next"],
        "request": ["what", "which", "could","address","phone number"],
        "negate": ["not", "don't", "do not", "nothing", "no", "never"],
        "confirm": ["yes"],
        "restart": ["restart"],
        "ack": ["ok", "okay"],
        "inform": ["i want"]
    }

    for intent, keywords in rules.items():
        for keyword in keywords:
            keyword_words = keyword.split()
            if len(keyword_words) == 1 and keyword_words[0] in utterance_words:
                return intent
            elif len(keyword_words) > 1:
                for i in range(len(utterance_words) - len(keyword_words) + 1):
                    if utterance_words[i:i+len(keyword_words)] == keyword_words:
                        return intent

    return majority_label



def calculate_majority_label_accuracy(dataset):
    correct = 0
    for i in range(len(dataset)):
        label = dataset[i][0]
        utterances = dataset[i][1]
        prediction = majority_baseline_predict(utterances)
        if prediction == label:
            correct += 1
    return correct / len(dataset)

def calculate_accuracy(dataset):
    correct = 0
    for i in range(len(dataset)):
        label = dataset[i][0]
        utterances = dataset[i][1]
        prediction = rule_based_predict(utterances)
        if prediction == label:
            correct += 1
    return correct / len(dataset)