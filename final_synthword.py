# -*- coding: utf-8 -*-
import os
import re
import nltk
import pyphen
import itertools
import collections



def determine_tense_input(sentence):
    text = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(text)
    tags = {word[0]:word[1] for word in tagged}
    tense = {}
    tense["future"] = len([word for word in tagged if word[1] == "MD"])
    tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
    tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]]) 
    return (tags,tense)



class Suggestor:
	# suggesr closest words to list of patterns
	def __init__(self, pattern):
		self.WORDS = collections.Counter(self.load_words(pattern)) # or use open(big.txt,..)
	
	def load_words(self, text):
		return re.findall(r'\w+', text.lower())
	
	def P(word=""): 
		# Probability of `word`.
		return self.WORDS[word] / sum(WORDS.values())
	
	def correction(self, word):
		# Most probable spelling correction for word.
		return max(self.candidates(word), key=self.P)#candidates(word)
	
	def candidates(self, word):
		# Generate possible spelling corrections for word.
		return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])
	
	def known(self, words):
		# The subset of `words` that appear in the dictionary of self.WORDS.
		return set(w for w in words if w in self.WORDS)
	
	def edits1(self, word):
		# All edits that are one edit away from `word`.
		letters	= 'abcdefghijklmnopqrstuvwxyz'
		splits	 = [(word[:i], word[i:])	for i in range(len(word) + 1)]
		deletes	= [L + R[1:]			   for L, R in splits if R]
		transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
		replaces   = [L + c + R[1:]		   for L, R in splits if R for c in letters]
		inserts	= [L + c + R			   for L, R in splits for c in letters]
		return set(deletes + transposes + replaces + inserts)
	
	def edits2(self, word):
		# All edits that are two edits away from `word`.
		return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
	
	def match(self, synth_word):
		return self.known(self.edits2(synth_word))



def sshapify(word):
	# transliterate letters to Sshapenmwiezphardt
	sshapenmwiezphardt = {
	"CH":"TY", "EE":"ΣI", "CC":"KС", "C":"K", "SC":"С", "Q":"KЦ", "X":"KС", # this line is unofficial but a remedy
	"A":"Λ", "a":"a", "E":"Σ", "e":"e", "I":"I", "i":"i", "U":"U", "u":"u", "O":"O", "o":"o", "B":"V",
	"b":"v", "D":"Δ", "d":"d", "F":"F", "f":"f", "G":"Γ", "g":"g", "H":"Х", "h":"h", "J":"J", "j":"j",
	"K":"K", "k":"k", "L":"L", "l":"l", "M":"M", "m":"μ", "N":"N", "n":"n", "P":"П", "p":"п", "R":"R",
	"r":"π", "S":"С", "s":"с", "T":"T", "t":"t", "W":"Ц", "w":"ц", "Y":"Y", "y":"y", "Z":"Z", "z":"z"
	}
	
	for item in sshapenmwiezphardt.items():
		word = word.replace(item[0],item[1])
	
	return word



hyph = pyphen.Pyphen(lang="en_US")


for word_pos in ("CC","CD","DT","EX","FW","IN","JJ","LS","MD","PDT","PRP","PRP$","RB","RBR","RBS","RP","TO","UH","WDT","WP","WP$","WRB","VB","NN"):
	with open("./sshapwords/{0}_Sshapen.words".format(word_pos), "a") as fiel:
		for origin in os.listdir("./transliterd/"):
			with open("./transliterd/" + origin) as f:
				words = [word.lower() for word in f.read().split("\n") if bool(word)]
				words.append(origin)
				words = sorted(words, key=len)
			
			try:
				pos_out = determine_tense_input(origin)
				if len(pos_out[0].keys()) == 1:
					if pos_out[0][origin] == word_pos:
						# suggestion reference object
						cmp_sound = Suggestor("\x20".join(words)) # origin: English
						correlates = [c for c in cmp_sound.match(origin)]
						correlates.append(words[0])
						sorted_words = sorted(set(correlates),key=len)
						
						# pick the shortest alternative
						hyph_word0 = hyph.inserted(sorted_words[0]).split("-")
						hyph_word1 = hyph.inserted(sorted_words[1]).split("-")
						w_distance = nltk.edit_distance(hyph_word0[0],hyph_word1[0])
						
						# get new word
						if len(sorted_words) < 1: # if there are only two choices.
							final_word = sorted_words[0]
						else:
							# get levenstein distance via list comprehension
							x_w = {} # weighted words
							for i in itertools.permutations(sorted_words, 2):
								if not i[0] in x_w:
									x_w[i[0]] = nltk.edit_distance(i[0],i[1])
								else:
									x_w[i[0]] += nltk.edit_distance(i[0],i[1])
							
							# final process 
							ww_words = sorted([word for word in x_w.items() if word[1] == min(x_w.values())], key=len)
							final_word = ww_words[0][0]
						
						# make suggestions
						#if w_distance < int(len(origin)/2): # - don't now which one is better
						if w_distance <= int(len(hyph_word1[0])/2):
							suggest = sorted_words[0]
						else:
							suggest = ""
						
						sshaped0 = sshapify(final_word.upper())
						sshaped1 = sshapify(suggest.upper())
						
						# empty `suggest` if `forged` has same spell 
						if sshaped0 == sshaped1:
							sshaped1 = ""
							suggest = ""
						
						new_data = {'origin':origin, 'forged':[final_word,sshaped0], 'suggest':[suggest,sshaped1], 'relatives':sorted_words}
						#print("{0}\n".format(new_data))
						fiel.write("{0}\n".format(new_data))
						#print("origin: ", origin, " => " , final_word, sshapify(final_word.upper()), sorted_words)
						
			except Exception as errr:
				alt_out = {'origin':origin, 'forged':[sorted_words[0],sshapify(sorted_words[0].upper())], 'suggest':['',''], 'relatives':sorted_words} # alternative output
				errr_msg = "{0}: {1}\n".format(errr, alt_out)
				#print(errr_msg)
				with open("./sshapwords/{0}_error_final_synthword.log".format(word_pos),"a") as errlog:
					errlog.write(errr_msg)



