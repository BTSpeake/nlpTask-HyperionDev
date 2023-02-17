import spacy 
nlp = spacy.load('en_core_web_sm')

### code example 1 
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))


### code example 2 
print() 

tokens = nlp("cat apple monkey banana")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

## It is interesting how it gives a high similarity with apple and banana, most likely due to the fact 
# that they are both fruits. However, cat and monkey have a comparatively very low score. One might assume 
# that they should have a fairly high score due to them both being animals/mammals. Clearly however, 
# the language model has identified that although they are both mammals, they are still very different 
# species. It is also interesting how monkey scores higher with both fruits than cat does, most likely due to 
# the higher association of monkeys eating fruits than cats do. 

### code example 3 
print() 

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car", 
             "I\'ve lost my car in my car", 
             "I\'d like my boat back",
             "I will name my dog Diana"
            ]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)



## When running with 'en_core_sm' it gives a warning that it cannot accurately identify similarities between individual words directly, 
# instead using other identifying factors which are more general. This has resulted in the overall scores for comparing single words to 
# be higher. However, it fails to identify the same level of similarity between the two fruits 'apple' and 'banana'. Additionally, it in 
# general results in lower scores when comparing the similarity of full sentences, presumably due to the more complex interplay between the 
# the words which are not well described by the more simple language package. 