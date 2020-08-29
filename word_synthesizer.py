#!/bin/python
# -*- coding: utf-8 -*-

import os
import re
import pyphen
from collections import Counter
from itertools import permutations
#from arabic_pronunciation import phonetise_Arabic
#phonetise_Arabic.phonetise_word("بِمُسْتَطِيل")


"""
https://youtu.be/0HW0q13kBPU
https://youtu.be/xvWBd4s_6zE
https://youtu.be/ZKkzEBtIoH8


Levenshtein: https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
word matching: \b(\w*work\w*)\b


# better one: https://en.wikipedia.org/wiki/Levenshtein_distance
# better one: https://en.wikipedia.org/wiki/N-gram#Google_use_of_N-gram


I used Predictive Search algorithm created by a Google software engineer.
I also made a Supervised Machine Learning algorithm for word syllabification.

Google
"""



class ML_hyph:
	def __init__(self):
		# set dataset name
		self.dataset_name = "dataset.ref"
		
		# set custom keyword input
		try:
			self.confirm = raw_input
		except:
			self.confirm = input
		
		# initialize dataset if it doesn't exist
		if not os.path.exists(self.dataset_name):
			dataset = """ams-ter-dam\nik\nmoy\nmyo\nist\nma-ga\nnosht\nek-wi-tor\nre-puv-lik\nu-ni-tyed\nis-le\nreyng-dom\nprin-si-pe\nklok\nstel-lar\nu-ni-ver-sed\nes-pa-syo\nv-tras\nsyn-ra\nman-ya-na\npo-ta-ta\na-to-mat\nfo-deu\ns-shlaf\nper-son\nal-lo\nde\nsagh\nvoir\nen-kryp\nde-kryp\nsyst\ngle-yo\npie-na\ns-shlert\nserd-ti\nspa-ko-nay\ngough\nda-pa-rae\ntan-to\nsi-ver\nda-shiur\na\no-ki\nka-lis-dad-sed\nsyen-sya\nflo-sfr\nsay-ko-lya\nsay-ko-lyast\na-ni-virs\na-ni-vir-sti\ns-sha-penm-wiez-phardt\ndog-go\nu-nik\no-ri-hi-nal\na-ban-do-ned\nab-so-lu-te\na-do-ra-ble\nad-ven-tu-rous\nac-cep-ta-ble\nac-clai-med\nac-com-pli-shed\nac-cu-ra-te\na-ching\na-cro-ba-tic\nac-ti-ve\nac-tual\na-dept\nad-mi-ra-ble\na-do-le-scent\nad-van-ced\naf-fec-tio-na-te\nag-gra-va-ting\nag-gres-si-ve\na-gi-le\na-go-ni-zing\na-gre-eable\na-lar-ming\na-lie-na-ted\nall\nal-truis-tic\nam-bi-ti-ous\nam-ple\nan-cho-red\nan-cient\nang-ry\nang-ui-shed\nan-nual\nan-ti-que\nan-xi-ous\nap-pre-hen-si-ve\nap-pro-pria-te\narc-tic\nas-su-red\nas-to-ni-shing\nat-ta-ched\nat-ten-ti-ve\nat-trac-ti-ve\naus-te-re\nau-then-tic\nau-tho-ri-zed\nau-to-ma-tic\na-va-ri-ci-ous\na-ve-ra-ge\naw-ful\nawk-ward\nba-by-ish\nbag-gy\nbar-ren\nbe-au-ti-ful\nbe-la-ted\nbe-ne-fi-cial\nbe-wit-ched\nbig-hear-ted\nbio-de-gra-da-ble\nbi-te-si-zed\nblack\nbla-ring\nble-ak\nbliss-ful\nblue\nblu-shing\nboi-ling\nbo-ring\nboun-cy\nboun-ti-ful\nbrea-ka-ble\nbril-liant\nbro-ken\nbron-ze\nbrui-sed\nbub-bly\nbur-den-so-me\nbust-ling\nbut-te-ry\ncal-cu-la-ting\nca-re-fre-e\nca-re-less\ncau-ti-ous\nca-ver-nous\nce-le-bra-ted\nche-er-ful\nche-e-ry\nchil-ly\ncir-cu-lar\nclear-cut\nclou-dy\nclue-less\nclut-te-red\ncoar-se\nco-lor-less\nco-los-sal\ncom-for-ta-ble\ncom-pas-sio-na-te\ncom-pe-tent\ncom-ple-te\ncom-plex\ncom-pli-ca-ted\ncon-cer-ned\ncon-sci-ous\ncon-si-de-ra-te\ncon-ven-tio-nal\nco-o-ked\nco-ol\nco-o-pe-ra-ti-ve\nco-or-di-na-ted\ncou-ra-ge-ous\ncour-te-ous\ncrea-ti-ve\ncro-o-ked\ncrow-ded\ncul-ti-va-ted\ncy-lin-dri-cal\ndan-ge-rous\ndazz-ling\ndea-fe-ning\nde-ci-si-ve\nde-fen-se-less\nde-fen-si-ve\nde-fiant\nde-fi-cient\nde-fi-ni-ti-ve\nde-lec-ta-ble\nde-li-ci-ous\nde-light-ful\nde-man-ding\ndes-crip-ti-ve\nde-tai-led\ndif-fe-rent\ndi-li-gent\ndim-wit-ted\ndi-sas-trous\ndown-right\ndis-gui-sed\ndis-tinct\ndry\ndull\neach\nea-ger\near-nest\near-ly\nea-sy-going\ne-di-ble\ne-la-bo-ra-te\nel-der-ly\ne-lec-tric\nel-lip-ti-cal\nem-bar-ras-sed\nem-bel-li-shed\ne-mo-tio-nal\nen-chan-ting\nen-ligh-te-ned\ne-nor-mous\ne-qual\ne-qua-to-rial\nes-sen-tial\nes-te-e-med\neu-pho-ric\ne-ver-gre-en\ne-ver-las-ting\nex-cel-lent\ne-xem-pla-ry\nex-ci-ting\nex-pen-si-ve\nex-pe-rien-ced\nex-tra-ne-ous\nex-tro-ver-ted\nex-tra-lar-ge\nex-tra-small\nfa-bu-lous\nfaith-ful\nfa-mous\nfar-off\nfa-vo-ra-ble\nfear-ful\nfear-less\nfirst-hand\nflam-bo-yant\nflaw-less\nflip-pant\nflu-ste-red\nfool-har-dy\nfo-o-lish\nforth-right\nfor-tu-na-te\nfre-e\nfre-quent\nfriend-ly\nfrigh-te-ned\nfrigh-te-ning\nfri-vo-lous\nfunc-tio-nal\ngar-gan-tuan\nga-se-ous\nge-nui-ne\nglass\nglea-ming\nglit-te-ring\nglo-ri-ous\ngor-ge-ous\ngran-dio-se\ngre-en\ngre-ga-ri-ous\ngro-tes-que\ngrou-chy\ngul-li-ble\nhap-py-go-luc-ky\nhard-to-find\nharm-less\nhar-mo-ni-ous\nhaun-ting\nheal-thy\nheart-felt\nhea-ven-ly\nhigh-le-vel\nhos-pi-ta-ble\nhu-mi-lia-ting\nic-ky\ni-dea-lis-tic\ni-dio-tic\nill-fa-ted\nill-in-for-med\nil-li-te-ra-te\nil-lus-tri-ous\ni-ma-gi-na-ti-ve\nim-ma-te-rial\nim-me-dia-te\nim-men-se\nim-pas-sio-ned\nim-pec-ca-ble\nim-par-tial\nim-per-fect\nim-per-tur-ba-ble\nim-prac-ti-cal\nim-pres-sio-na-ble\nim-pres-si-ve\nin-com-pa-ra-ble\nin-com-ple-te\nin-con-se-quen-tial\ni-nex-pe-rien-ced\nin-fa-mous\nin-fa-tua-ted\nin-no-cent\nin-sig-ni-fi-cant\nin-struc-ti-ve\nin-sub-stan-tial\nin-tel-li-gent\nin-ten-tio-nal\nin-ter-na-tio-nal\nir-res-pon-si-ble\nir-ri-ta-ting\njea-lous\nka-lei-do-sco-pic\nkind-hear-ted\nknow-led-gea-ble\nko-o-ky\nlight-hear-ted\nli-ka-ble\nlit-tle\nloath-so-me\nlo-o-se\nlus-trous\nmar-ried\nmas-si-ve\nmea-ger\nme-dio-cre\nmi-nia-tu-re\nmi-ser-ly\nmons-trous\nmonth-ly\nmo-nu-men-tal\nmo-tion-less\nmoun-tai-nous\nmuf-fled\nmul-ti-co-lo-red\nmys-te-ri-ous\nnau-ti-cal\nne-ces-sa-ry\nneigh-bo-ring\nno-te-wor-thy\no-be-dient\noc-ca-sio-nal\nodd-ball\noff-beat\nof-fi-cial\nold-fa-shio-ned\nop-ti-mis-tic\nour\nout-lying\nout-going\nout-lan-dish\nout-ra-ge-ous\nout-stan-ding\no-ver-co-o-ked\no-ver-due\npas-sio-na-te\npes-si-mis-tic\nphy-si-cal\nplain-ti-ve\npoint-less\npres-ti-gi-ous\npric-kly\npri-va-te\npro-ba-ble\npro-duc-ti-ve\npro-fi-ta-ble\nqua-int\nqua-li-fied\nquar-rel-so-me\nquar-ter-ly\nques-tio-na-ble\nquick-wit-ted\nqu-iet\nquin-tes-sen-tial\nquiz-zi-cal\nrec-tan-gu-lar\nrea-lis-tic\nrea-so-na-ble\nres-pect-ful\nscho-lar-ly\nscien-ti-fic\nscorn-ful\nscrat-chy\nscraw-ny\nse-cond-hand\nself-as-su-red\nself-re-liant\nsen-ti-men-tal\nsha-me-less\nshort-term\nshrill\nsim-plis-tic\nsni-ve-ling\nsphe-ri-cal\nso-phis-ti-ca-ted\nsor-row-ful\nspark-ling\nspec-ta-cu-lar\nsplen-did\nsqua-re\nsque-a-ky\nsquig-gly\nsti-mu-la-ting\nstra-ight\nstran-ge\nstrict\nstri-dent\nstri-ped\nstu-pen-dous\nsub-mis-si-ve\nsub-stan-tial\nsu-per-fi-cial\nsup-por-ti-ve\nsu-re-fo-o-ted\nsym-pa-the-tic\ntat-te-red\ntho-ught-ful\nthread-ba-re\nthun-de-rous\ntrau-ma-tic\ntri-vial\ntrust-wor-thy\nu-nac-cep-ta-ble\nun-com-mon\nun-con-sci-ous\nun-for-tu-na-te\nun-hap-py\nun-heal-thy\nu-ni-que\nun-na-tu-ral\nun-plea-sant\nun-rea-lis-tic\nun-sightly\nun-stea-dy\nun-tried\nun-true\nu-nu-sual\nun-wiel-dy\nun-writ-ten\nu-se-less\nut-ter\nve-ri-fia-ble\nvil-lai-nous\nwa-ter-log-ged\nwe-e\nwe-ek-ly\nwell-do-cu-men-ted\nwell-gro-o-med\nwell-in-for-med\nwell-lit\nwell-ma-de\nwell-off\nwi-de-e-yed\nwor-ri-so-me\nworth-less\nworth-whi-le"""
			with open(self.dataset_name,"wb") as f:
				f.write(dataset.encode()) # sample data for starter
		# read dataset
		with open(self.dataset_name,"rb") as f:
			self.dataset = f.read().decode().split("\n")
	
	
	def cryptify(self,data):
		# re-assign var for sanitized data
		data = list(data.replace("-","").split("\n")[0])
		
		# set lambda function for symbol determiner
		replacewhichtype = lambda v,c,d: v.lower() if re.match(r"[AEIUOaeiuo]",d) else c.lower() if re.match(r"[b-df-hj-np-tv-z]|[B-DF-HJ-NP-TV-Z]",d) else ""
		
		# flip repeats for every repetition
		last_cur = ""
		
		# iterate through data length
		for i in range(len(data)):
			if data[i] == last_cur:
				# re-assign + replace with (clone) symbol type
				data[i] = replacewhichtype("x","z",data[i])
				data[i-1] = data[i]
			else:
				# set another character as last reference
				last_cur = data[i]
				
				# re-assign + replace with (normal) symbol type
				data[i] = replacewhichtype("v","c",data[i])
		
		# return bundled (cryptified) data
		return "".join(data)
	
	
	def hyphenate(self,word):
		# cryptify reference word
		wordcrpt = self.cryptify(word)
		
		# re-assign var for sanitized word
		word = list(word.replace("-","").split("\n")[0])
		
		# iterate through dataset patterns
		for ds in self.dataset:
			# if matched then this is the possible hyphenation style; hyphenate now then break
			if self.cryptify(ds) == wordcrpt:
				# hyphenate word
				for i in range(len(ds)):
					# insert "-" to specific index in word
					if ds[i] == "-":
						word.insert(i,"-")
				
				# we've found the answer, so..
				break
		
		# return bundled (hyphenated) word
		return "".join(word)
	
	
	def train(self,word):
		# make dataset_crpt for pattern structure checking
		dataset_crpt = [self.cryptify(ds_crpt) for ds_crpt in self.dataset if ds_crpt!=""]
		
		while 1:
			suggest = self.confirm("\nNew pattern! '{0}', is this correct? (Y/suggest another) ".format(word))
			if not bool(suggest):
				break
			
			#word = suggest if (bool(suggest) and not bool(confirm)) else word
			word = suggest
			
			confirm = self.confirm("\nSave '{0}' ::: Are you sure? (Y/n) ".format(word))
			
			
			if not bool(confirm):
				break
			
			print("repeating.....")
		
		# update dataset if it doesn't match on `dataset_crpt` patterns
		if not self.cryptify(word) in dataset_crpt:
			self.dataset.append(word)
			with open(self.dataset_name,"ab") as f:
				f.write("\n{0}".format(word).encode())
		
		# return word
		return word



class Suggestor:
	# suggesr closest words to list of patterns
	def __init__(self, pattern):
		self.WORDS = Counter(self.load_words(pattern)) # or use open(big.txt,..)
	
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



def findCenter(dlist):
	# find semi-/center of the list
	dlist = sorted(dlist)
	
	if len(dlist)==0:
		return dlist
	elif len(dlist)%2==0:
		return dlist[int(len(dlist)/2)-1]
	elif len(dlist)%2==1:
		return dlist[int(len(dlist)/2)]



def sshapify(word):
	# transliterate letters to Sshapenmwiezphardt
	sshapenmwiezphardt = {
	"EE":"ΣI", "CC":"KС", "C":"K", "SC":"С", "Q":"KЦ", "X":"KС", # this line is unofficial but a remedy
	"A":"Λ", "a":"a", "E":"Σ", "e":"e", "I":"I", "i":"i", "U":"U", "u":"u", "O":"O", "o":"o", "B":"V",
	"b":"v", "D":"Δ", "d":"d", "F":"F", "f":"f", "G":"Γ", "g":"g", "H":"Х", "h":"h", "J":"J", "j":"j",
	"K":"K", "k":"k", "L":"L", "l":"l", "M":"M", "m":"μ", "N":"N", "n":"n", "P":"П", "p":"п", "R":"R",
	"r":"π", "S":"С", "s":"с", "T":"T", "t":"t", "W":"Ц", "w":"ц", "Y":"Y", "y":"y", "Z":"Z", "z":"z"
	}
	
	for item in sshapenmwiezphardt.items():
		word = word.replace(item[0],item[1])
	
	return word



def syllabify(word):#[ok] but change this to self-made algorithm
	# get word syllable, use my version of syllabifier for reliability
	syllabfd = SML_hyph.hyphenate(word)
	
	# if no 'hyphens' and word is longer than 3 (and not v..c)
	if not "-" in syllabfd and len(word) > 3:
		syllabfd = pyphen.Pyphen(lang='it_IT',cache=False,left=1,right=1).inserted(word)
		syllabfd = SML_hyph.train(syllabfd)
	
	print("SYLLABLES",syllabfd)
	
	return syllabfd.split("-")



def mix_syllab(standard="",syllabs={},origins={}): # syllabs/syllabs/words
	# sort syllables
	syllabs = set(sorted(syllabs,key=len))
	
	# sort origins
	origins = set(sorted(origins,key=len))
	
	# syllables* length
	block=len(syllabs)
	
	## min and max syllabic length standards ##
	# standard word length
	std_A = len(syllabify(standard))
	
	# first item from words (lowest len amongst mean)
	std_B = len(syllabify(next(iter(origins))))
	
	# min-max syllabic length, sort lowest to highest length
	MinMax = sorted([std_A,std_B])
	#print(std_A,std_B)
	
	# suggestion reference object
	cmp_sound = Suggestor("\x20".join(origins))
	print("origin",origins)
	
	# sshap. synthetic words
	formed_words = {}
	
	# mix syllables to form valid words
	for i in range(MinMax[0], MinMax[1] + 1):
		print("correlates: ",syllabs)
		# Get all permutations of length-range i
		for synth_word in list(permutations(sorted(syllabs),i)):
			# find possible candidates
			synth_word = "".join(synth_word)
			print("aaa",synth_word)
			correlates = [c for c in cmp_sound.match(synth_word)]
			
			#print("correlates: ",correlates, syllabs)
			if bool(correlates):
				formed_words[synth_word] = correlates
				#print("\n[CORRELATION FOUND!]\n:: processing: '{0}'\n:: correlates: {1}\n:: found: {2}\n:: total found: {3}\n".format(synth_word,correlates,formed_words.keys(),len(formed_words)))
			else:
				#print("processing: '{0}'.... no correlation.".format(synth_word))
				pass
	
	print("done processing! synthetic words: {0}".format(len(formed_words)))
	return formed_words



# PLAN A (best) [main entry point]
def feeder_synthesiser(words=[]):
	syllabs = set() # syllabified chosen reference ; to be fed into mixer
	
	# re-sort words
	words = set(sorted(words,key=len))
	
	# find standard (ave length) word
	standard = findCenter(words)
	
	# non-repeating word queries in different language
	for word in words:
		syllables = syllabify(word) # split to syllables
		syllabs |= set(syllables) # update reference syllabs
	
	# mix * syllables with current syllabs ; results: {synth_word:[correlates,..]}
	results = mix_syllab(standard=standard, syllabs=syllabs, origins=words)
	#print("AAAAAAAAAAAA: ",results)
	
	# set correlation ranking
	correlates_rank = {}
	
	# rank correlations
	for result in results.items():
		for e_correlates in result[1]:
			# create initial record if not indexed
			if not e_correlates in correlates_rank:
				correlates_rank[e_correlates] = 0
			
			# increment record by 1
			correlates_rank[e_correlates] += 1
	
	# sort `correlates_rank` (in ascending order)
	correlates_rank = {k: v for k, v in sorted(correlates_rank.items(), key=lambda val: val[1])} # by value
	correlates_rank = {k: v for k, v in sorted(correlates_rank.items(), key=lambda key: key[0])} # by key
	
	
	#####################################
	# major words and minimum counts
	#####################################
	
	
	# get the minimum value length
	min_val = len(sorted(correlates_rank,key=len)[0])
	
	# collect specific items with respect to `min_val`
	min_synthwords = {items[0]:items[1] for items in correlates_rank.items() if len(items[0]) == min_val}
	
	# set maximum min_synthwords' count
	max_val = max(min_synthwords.items(),key=lambda val:val[1])[1]
	
	# collect specific items with respect to `max_val`
	major_synthwords = [items[0] for items in correlates_rank.items() if items[1] == max_val]
	
	#[???] get the minimum value length
	synthword = sorted(major_synthwords,key=len)
	
	# get final output from results
	if len(synthword):
		return synthword[0]
	else:
		return standard



# set ML hyphenation object
SML_hyph = ML_hyph()


if __name__ == "__main__":
	#synthword = feeder_synthesiser(words=["alesana","alessana","alessanna","alesanna"])
	#print(synthword)
	
	for fls in os.listdir("./transliterd"):
		try:
			sanitized_words = []
			with open("./transliterd/"+fls) as f1:
				word_d = f1.read().split("\n")
			
			for word in word_d:
				current_word = ""
				for lettr in list(word):
					if lettr.isalpha():
						current_word += lettr
				
				if bool(current_word):
					sanitized_words.append(current_word.lower())
			
			if bool(sanitized_words):
				print(sanitized_words)
				synthword = feeder_synthesiser(words=sanitized_words)
				print("final: {0}".format(synthword))
			
		except Exception as er:
			print("error: ", er)


"""
import numpy
import matplotlib.pyplot as plt
# Add title and axis names
plt.title('{0}'.format("experimental data synthesis (Sshapenmwiezphardt)"))#final_name_after_creation"))
plt.xlabel('iterations')
plt.ylabel('candidates expansion')
# axis scatter plot data
x=[a[i][0] for i in range(len(a))]
y=[a[i][1] for i in range(len(a))]
x = numpy.array(x)
y = numpy.array(y)
plt.plot(x, y, 'o')
# linear regression line
m, b = numpy.polyfit(x, y, 1)
# compile plot
plt.plot(x, m*x + b)
# show plot
plt.show()
"""


"""
based_output_length = [(base if (block-(base*i))>base else (block-(base*i))) for i in range(block//base+block%base) if (block-(base*i))>0]


dic = pyphen.Pyphen(lang='it_IT',cache=False,left=1,right=1)
return dic.inserted(word).split("-")


#results = mix_syllab(standard={"a","le"},syllabs=["a","le","sa","na","w","f"],origins={"alezana","alesauna","alesaanaa","alesaana"})
#mix_syllab(standard="alesala",syllabs={"a", "le", "sa", "na", "san", "xa", "ma", "la"},origins={"alesanxa", "alesana", "alesala", "alexana", "alexama"})
#mix_syllab(standard={"a","le","sa","na"},syllabs={"a", "le", "sa", "na", "xa"},origins={"alesana", "alexana", })
"""

