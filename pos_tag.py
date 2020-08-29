from nltk import word_tokenize, pos_tag

def determine_tense_input(sentence):
    text = word_tokenize(sentence)
    tagged = pos_tag(text)
    
    tags = {word[0]:word[1] for word in tagged}
    
    tense = {}
    tense["future"] = len([word for word in tagged if word[1] == "MD"])
    tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
    tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]]) 
    return (tags,tense)


print(determine_tense_input("this"))
