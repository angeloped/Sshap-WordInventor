import pyphen

"""
possible hyphenators: (by rank)
1. french
2. polish
"""

for lang in pyphen.LANGUAGES:
	print("\n\n\nLANGUAGE: {0}".format(lang))
	dic = pyphen.Pyphen(lang=lang,cache=False,left=1,right=1)
	print(dic.inserted('alettergrephendt'))
	print(dic.inserted('sshapenmwiezphardt'))
	print(dic.inserted('alesana'))
	print(dic.inserted('sshlert'))
	print(dic.inserted('flavour'))
	print(dic.inserted('alita'))
	print(dic.inserted('alessa'))
	print(dic.inserted('alesa'))
	print(dic.inserted('ale'))
	print(dic.inserted('akekokaooka'))
	print(dic.inserted('lode'))
	print(dic.inserted('lodelodelede'))
	print(dic.inserted('vtras'))
	print(dic.inserted('avelin'))
	print(dic.inserted('aveline'))
	print(dic.inserted('loo'))
	print(dic.inserted('look'))
	print(dic.inserted('egol'))
	print(dic.inserted('estetik'))
	print(dic.inserted('estetik'))
	print(dic.inserted('isetik'))



"""
# I used this when creating train data
dic = pyphen.Pyphen(lang="it_IT",cache=False,left=1,right=1)
with open("data") as f,open("data2","a") as ff:
	aa = f.read().split("\n")
	for a in aa:
		ff.write(dic.inserted(a)+"\n")
"""
