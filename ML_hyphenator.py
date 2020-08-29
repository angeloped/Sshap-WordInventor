#!/usr/bin/python
import os
import re
import sys


"""
name: ML_hyphenator.py
project: Sshapenmwiezphardt language synthesiser
author: Bryan Angelo
description: A simple (Supervised) machine learning algorithm for processing
             Syllables of the synthetic language named S-sha-penm-wiez-phardt.

misc snippets: (useful someday)
vowel: ([AEIUOaeiuo]+?)\1+
cons: ([b-df-hj-np-tv-z]|[B-DF-HJ-NP-TV-Z])\1+
"""

dataset = """ams-ter-dam\nik\nmoy\nmyo\nist\nma-ga\nnosht\nek-wi-tor\nre-puv-lik\nu-ni-tyed\nis-le\nreyng-dom\nprin-si-pe\nklok\nstel-lar\nu-ni-ver-sed\nes-pa-syo\nv-tras\nsyn-ra\nman-ya-na\npo-ta-ta\na-to-mat\nfo-deu\ns-shlaf\nper-son\nal-lo\nde\nsagh\nvoir\nen-kryp\nde-kryp\nsyst\ngle-yo\npie-na\ns-shlert\nserd-ti\nspa-ko-nay\ngough\nda-pa-rae\ntan-to\nsi-ver\nda-shiur\na\no-ki\nka-lis-dad-sed\nsyen-sya\nflo-sfr\nsay-ko-lya\nsay-ko-lyast\na-ni-virs\na-ni-vir-sti\ns-sha-penm-wiez-phardt\ndog-go\nu-nik\no-ri-hi-nal\na-ban-do-ned\nab-so-lu-te\na-do-ra-ble\nad-ven-tu-rous\nac-cep-ta-ble\nac-clai-med\nac-com-pli-shed\nac-cu-ra-te\na-ching\na-cro-ba-tic\nac-ti-ve\nac-tual\na-dept\nad-mi-ra-ble\na-do-le-scent\nad-van-ced\naf-fec-tio-na-te\nag-gra-va-ting\nag-gres-si-ve\na-gi-le\na-go-ni-zing\na-gre-eable\na-lar-ming\na-lie-na-ted\nall\nal-truis-tic\nam-bi-ti-ous\nam-ple\nan-cho-red\nan-cient\nang-ry\nang-ui-shed\nan-nual\nan-ti-que\nan-xi-ous\nap-pre-hen-si-ve\nap-pro-pria-te\narc-tic\nas-su-red\nas-to-ni-shing\nat-ta-ched\nat-ten-ti-ve\nat-trac-ti-ve\naus-te-re\nau-then-tic\nau-tho-ri-zed\nau-to-ma-tic\na-va-ri-ci-ous\na-ve-ra-ge\naw-ful\nawk-ward\nba-by-ish\nbag-gy\nbar-ren\nbe-au-ti-ful\nbe-la-ted\nbe-ne-fi-cial\nbe-wit-ched\nbig-hear-ted\nbio-de-gra-da-ble\nbi-te-si-zed\nblack\nbla-ring\nble-ak\nbliss-ful\nblue\nblu-shing\nboi-ling\nbo-ring\nboun-cy\nboun-ti-ful\nbrea-ka-ble\nbril-liant\nbro-ken\nbron-ze\nbrui-sed\nbub-bly\nbur-den-so-me\nbust-ling\nbut-te-ry\ncal-cu-la-ting\nca-re-fre-e\nca-re-less\ncau-ti-ous\nca-ver-nous\nce-le-bra-ted\nche-er-ful\nche-e-ry\nchil-ly\ncir-cu-lar\nclear-cut\nclou-dy\nclue-less\nclut-te-red\ncoar-se\nco-lor-less\nco-los-sal\ncom-for-ta-ble\ncom-pas-sio-na-te\ncom-pe-tent\ncom-ple-te\ncom-plex\ncom-pli-ca-ted\ncon-cer-ned\ncon-sci-ous\ncon-si-de-ra-te\ncon-ven-tio-nal\nco-o-ked\nco-ol\nco-o-pe-ra-ti-ve\nco-or-di-na-ted\ncou-ra-ge-ous\ncour-te-ous\ncrea-ti-ve\ncro-o-ked\ncrow-ded\ncul-ti-va-ted\ncy-lin-dri-cal\ndan-ge-rous\ndazz-ling\ndea-fe-ning\nde-ci-si-ve\nde-fen-se-less\nde-fen-si-ve\nde-fiant\nde-fi-cient\nde-fi-ni-ti-ve\nde-lec-ta-ble\nde-li-ci-ous\nde-light-ful\nde-man-ding\ndes-crip-ti-ve\nde-tai-led\ndif-fe-rent\ndi-li-gent\ndim-wit-ted\ndi-sas-trous\ndown-right\ndis-gui-sed\ndis-tinct\ndry\ndull\neach\nea-ger\near-nest\near-ly\nea-sy-going\ne-di-ble\ne-la-bo-ra-te\nel-der-ly\ne-lec-tric\nel-lip-ti-cal\nem-bar-ras-sed\nem-bel-li-shed\ne-mo-tio-nal\nen-chan-ting\nen-ligh-te-ned\ne-nor-mous\ne-qual\ne-qua-to-rial\nes-sen-tial\nes-te-e-med\neu-pho-ric\ne-ver-gre-en\ne-ver-las-ting\nex-cel-lent\ne-xem-pla-ry\nex-ci-ting\nex-pen-si-ve\nex-pe-rien-ced\nex-tra-ne-ous\nex-tro-ver-ted\nex-tra-lar-ge\nex-tra-small\nfa-bu-lous\nfaith-ful\nfa-mous\nfar-off\nfa-vo-ra-ble\nfear-ful\nfear-less\nfirst-hand\nflam-bo-yant\nflaw-less\nflip-pant\nflu-ste-red\nfool-har-dy\nfo-o-lish\nforth-right\nfor-tu-na-te\nfre-e\nfre-quent\nfriend-ly\nfrigh-te-ned\nfrigh-te-ning\nfri-vo-lous\nfunc-tio-nal\ngar-gan-tuan\nga-se-ous\nge-nui-ne\nglass\nglea-ming\nglit-te-ring\nglo-ri-ous\ngor-ge-ous\ngran-dio-se\ngre-en\ngre-ga-ri-ous\ngro-tes-que\ngrou-chy\ngul-li-ble\nhap-py-go-luc-ky\nhard-to-find\nharm-less\nhar-mo-ni-ous\nhaun-ting\nheal-thy\nheart-felt\nhea-ven-ly\nhigh-le-vel\nhos-pi-ta-ble\nhu-mi-lia-ting\nic-ky\ni-dea-lis-tic\ni-dio-tic\nill-fa-ted\nill-in-for-med\nil-li-te-ra-te\nil-lus-tri-ous\ni-ma-gi-na-ti-ve\nim-ma-te-rial\nim-me-dia-te\nim-men-se\nim-pas-sio-ned\nim-pec-ca-ble\nim-par-tial\nim-per-fect\nim-per-tur-ba-ble\nim-prac-ti-cal\nim-pres-sio-na-ble\nim-pres-si-ve\nin-com-pa-ra-ble\nin-com-ple-te\nin-con-se-quen-tial\ni-nex-pe-rien-ced\nin-fa-mous\nin-fa-tua-ted\nin-no-cent\nin-sig-ni-fi-cant\nin-struc-ti-ve\nin-sub-stan-tial\nin-tel-li-gent\nin-ten-tio-nal\nin-ter-na-tio-nal\nir-res-pon-si-ble\nir-ri-ta-ting\njea-lous\nka-lei-do-sco-pic\nkind-hear-ted\nknow-led-gea-ble\nko-o-ky\nlight-hear-ted\nli-ka-ble\nlit-tle\nloath-so-me\nlo-o-se\nlus-trous\nmar-ried\nmas-si-ve\nmea-ger\nme-dio-cre\nmi-nia-tu-re\nmi-ser-ly\nmons-trous\nmonth-ly\nmo-nu-men-tal\nmo-tion-less\nmoun-tai-nous\nmuf-fled\nmul-ti-co-lo-red\nmys-te-ri-ous\nnau-ti-cal\nne-ces-sa-ry\nneigh-bo-ring\nno-te-wor-thy\no-be-dient\noc-ca-sio-nal\nodd-ball\noff-beat\nof-fi-cial\nold-fa-shio-ned\nop-ti-mis-tic\nour\nout-lying\nout-going\nout-lan-dish\nout-ra-ge-ous\nout-stan-ding\no-ver-co-o-ked\no-ver-due\npas-sio-na-te\npes-si-mis-tic\nphy-si-cal\nplain-ti-ve\npoint-less\npres-ti-gi-ous\npric-kly\npri-va-te\npro-ba-ble\npro-duc-ti-ve\npro-fi-ta-ble\nqua-int\nqua-li-fied\nquar-rel-so-me\nquar-ter-ly\nques-tio-na-ble\nquick-wit-ted\nqu-iet\nquin-tes-sen-tial\nquiz-zi-cal\nrec-tan-gu-lar\nrea-lis-tic\nrea-so-na-ble\nres-pect-ful\nscho-lar-ly\nscien-ti-fic\nscorn-ful\nscrat-chy\nscraw-ny\nse-cond-hand\nself-as-su-red\nself-re-liant\nsen-ti-men-tal\nsha-me-less\nshort-term\nshrill\nsim-plis-tic\nsni-ve-ling\nsphe-ri-cal\nso-phis-ti-ca-ted\nsor-row-ful\nspark-ling\nspec-ta-cu-lar\nsplen-did\nsqua-re\nsque-a-ky\nsquig-gly\nsti-mu-la-ting\nstra-ight\nstran-ge\nstrict\nstri-dent\nstri-ped\nstu-pen-dous\nsub-mis-si-ve\nsub-stan-tial\nsu-per-fi-cial\nsup-por-ti-ve\nsu-re-fo-o-ted\nsym-pa-the-tic\ntat-te-red\ntho-ught-ful\nthread-ba-re\nthun-de-rous\ntrau-ma-tic\ntri-vial\ntrust-wor-thy\nu-nac-cep-ta-ble\nun-com-mon\nun-con-sci-ous\nun-for-tu-na-te\nun-hap-py\nun-heal-thy\nu-ni-que\nun-na-tu-ral\nun-plea-sant\nun-rea-lis-tic\nun-sightly\nun-stea-dy\nun-tried\nun-true\nu-nu-sual\nun-wiel-dy\nun-writ-ten\nu-se-less\nut-ter\nve-ri-fia-ble\nvil-lai-nous\nwa-ter-log-ged\nwe-e\nwe-ek-ly\nwell-do-cu-men-ted\nwell-gro-o-med\nwell-in-for-med\nwell-lit\nwell-ma-de\nwell-off\nwi-de-e-yed\nwor-ri-so-me\nworth-less\nworth-whi-le"""

try:
	confirm = raw_input
except:
	confirm = input


# initialize dataset if it doesn't exist
if not os.path.exists("dataset.ref"):
	with open("dataset.ref","wb") as f:
		f.write(dataset.encode()) # sample data for starter


# read dataset
with open("dataset.ref","rb") as f:
	dataset = f.read().decode().split("\n")


def cryptify(data):
	# re-assign var for sanitized data
	data = list(data.replace("-","").split("\n")[0])
	
	# set lambda function for case setter ; remove case sensitivity[ok]
	#setcase = lambda c,d: c.upper() if d.isupper() else c.lower()
	
	# set lambda function for symbol determiner
	#replacewhichtype = lambda v,c,d: setcase(v,d) if re.match(r"[AEIUOaeiuo]",d) else setcase(c,d) if re.match(r"[b-df-hj-np-tv-z]|[B-DF-HJ-NP-TV-Z]",d) else ""
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



def hyphenate(word):
	# cryptify reference word
	wordcrpt = cryptify(word)
	
	# re-assign var for sanitized word
	word = list(word.replace("-","").split("\n")[0])
	
	# iterate through dataset patterns
	for ds in dataset:
		# if matched then this is the possible hyphenation style; hyphenate now then break
		if cryptify(ds) == wordcrpt:
			# hyphenate word
			for i in range(len(ds)):
				# insert "-" to specific index in word
				if ds[i] == "-":
					word.insert(i,"-")
			
			break # we've found the answer, so..
		
	# return bundled (hyphenated) word
	return "".join(word)


help = "usage: python {0} <option> <your_dataset(s)>\n\noptions:\n--help - show this help.\n--train - train the program.\n--unify - unify multiple dataset. [python {0} --unify d1+d2+d3 outname]".format(sys.argv[0])


if __name__ == "__main__":
	if len(sys.argv) <= 4 and len(sys.argv) >= 3:
		#print(sys.argv)
		if sys.argv[1] == "--help":
			print(help)
		elif sys.argv[1] == "--train":
			# make dataset_crpt for pattern structure checking
			dataset_crpt = [cryptify(ds_crpt) for ds_crpt in dataset if ds_crpt!=""]
			old_data_len = len(dataset)
			
			# read dataset to feed them into main dataset and update
			with open(sys.argv[2],"rb") as fr, open("dataset.ref","wb") as fw:
				# update dataset with new unique patterns
				for ds_crpt in fr.read().decode().split("\n"):
					if ds_crpt!="" and not cryptify(ds_crpt) in dataset_crpt:
						print("data for: [{0}/{1}] = [{2}]\n".format(ds_crpt.replace("-",""),cryptify(ds_crpt),ds_crpt))
						dataset.append(ds_crpt)
						dataset_crpt.append(cryptify(ds_crpt))
					
					
					"""
					if ds_crpt!="" and not cryptify(ds_crpt) in dataset_crpt:
						# is this correct? are you sure? (real training)
						while 1:
							print("\ndata for: [{0}/{1}] = [{2}]".format(ds_crpt.replace("-",""),cryptify(ds_crpt),ds_crpt))
							confirm1 = confirm("is this correct? '{0}' (Y/n)".format(ds_crpt))
							confirm2 = confirm("are you sure? (Y/n)".format(ds_crpt))
							if confirm1=="" and confirm2=="":
								dataset.append(ds_crpt)
								dataset_crpt.append(cryptify(ds_crpt))
								break
							elif confirm1!="" and confirm2=="":
								break
							else:
								continue
					"""
				
				# update dataset file with new pattern
				fw.write("\n".join(dataset).encode())
			
			new_data_len = len(dataset)
			print("training done. {0} new added.".format(new_data_len-old_data_len))
			
		elif sys.argv[1] == "--unify" and len(sys.argv) == 4:
			unified_dataset = set()
			
			# update unified_dataset
			for dataset_name in sys.argv[2].split("+"):
				with open(dataset_name,"rb") as f:
					unified_dataset |= {data for data in f.read().split("\n") if data!=""}
			
			# sort and save unified dataset
			with open(sys.argv[3],"wb") as f:
				f.write("\n".join(sorted(unified_dataset)).encode())
			
			print("unification done.")
		
	elif len(sys.argv) == 2:
		print("[note]: this argument is reserved for training.\n\n{0}\n".format(help))
		print("output: {0}".format(hyphenate(sys.argv[1])))
	else:
		test = "angtasXas"
		print(cryptify("Laga-Ska"))
		print("Example output: {0} = {1}\n\ntype --help for more.".format(test,hyphenate(test))) # sample... you edit this.



