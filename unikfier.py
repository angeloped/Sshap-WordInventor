import pyphen
import ML_hyphenator

uniks = []
unik_sig = []


with open("words","rb") as f:
	wordlines = f.read().decode().split("\n")


dic = pyphen.Pyphen(lang="it_IT",cache=False,left=1,right=1)
for word in wordlines:
	wordsig = ML_hyphenator.cryptify(word)
	if not wordsig in unik_sig:
		uniks.append(dic.inserted(word))
		unik_sig.append(wordsig)


with open("words_x","wb") as f:
	f.write("\n".join(uniks))
