import os
import re

# cache of per-file HyphDict objects
hdcache = {}

# precompile some stuff
parse_hex = re.compile(r'\^{2}([0-9a-f]{2})').sub
parse = re.compile(r'(\d?)(\D?)').findall

try:
	from pkg_resources import resource_filename
	dictionaries_root = resource_filename('pyphen', 'dictionaries')
except ImportError:
	dictionaries_root = os.path.join(os.path.dirname(__file__), 'dictionaries')

LANGUAGES = {}
for filename in sorted(os.listdir(dictionaries_root)):
	if filename.endswith('.dic'):
		name = filename[5:-4]
		full_path = os.path.join(dictionaries_root, filename)
		LANGUAGES[name] = full_path
		short_name = name.split('_')[0]
		if short_name not in LANGUAGES:
			LANGUAGES[short_name] = full_path


def language_fallback(language):
	parts = language.replace('-', '_').split('_')
	while parts:
		language = '_'.join(parts)
		if language in LANGUAGES:
			return language
		parts.pop()


class AlternativeParser(object):
	def __init__(self, pattern, alternative):
		alternative = alternative.split(',')
		self.change = alternative[0]
		self.index = int(alternative[1])
		self.cut = int(alternative[2])
		if pattern.startswith('.'):
			self.index += 1

	def __call__(self, value):
		self.index -= 1
		value = int(value)
		if value & 1:
			return DataInt(value, (self.change, self.index, self.cut))
		else:
			return value


class DataInt(int):
	def __new__(cls, value, data=None, reference=None):
		obj = int.__new__(cls, value)
		if reference and isinstance(reference, DataInt):
			obj.data = reference.data
		else:
			obj.data = data
		return obj


class HyphDict(object):
	def __init__(self, filename):
		self.patterns = {}
		
		with open(filename, 'rb') as stream:
			# see "man 4 hunspell", iscii-devanagari is not supported by python
			charset = stream.readline().strip().decode('ascii')
			if charset.lower() == 'microsoft-cp1251':
				charset = 'cp1251'
			for pattern in stream:
				pattern = pattern.decode(charset).strip()
				if not pattern or pattern.startswith(('%', '#', 'LEFTHYPHENMIN', 'RIGHTHYPHENMIN','COMPOUNDLEFTHYPHENMIN', 'COMPOUNDRIGHTHYPHENMIN')):
					continue

				# replace ^^hh with the real character
				pattern = parse_hex(
					lambda match: chr(int(match.group(1), 16)), pattern)

				# read nonstandard hyphen alternatives
				if '/' in pattern:
					pattern, alternative = pattern.split('/', 1)
					factory = AlternativeParser(pattern, alternative)
				else:
					factory = int

				tags, values = zip(*[
					(string, factory(i or '0'))
					for i, string in parse(pattern)])

				# if only zeros, skip this pattern
				if max(values) == 0:
					continue

				# chop zeros from beginning and end, and store start offset
				start, end = 0, len(values)
				while not values[start]:
					start += 1
				while not values[end - 1]:
					end -= 1

				self.patterns[''.join(tags)] = start, values[start:end]

		self.cache = {}
		self.maxlen = max(len(key) for key in self.patterns)

	def positions(self, word):
		word = word.lower()
		points = self.cache.get(word)
		if points is None:
			pointed_word = '.%s.' % word
			references = [0] * (len(pointed_word) + 1)

			for i in range(len(pointed_word) - 1):
				for j in range(
						i + 1, min(i + self.maxlen, len(pointed_word)) + 1):
					pattern = self.patterns.get(pointed_word[i:j])
					if pattern:
						offset, values = pattern
						slice_ = slice(i + offset, i + offset + len(values))
						references[slice_] = map(
							max, values, references[slice_])

			points = [
				DataInt(i - 1, reference=reference)
				for i, reference in enumerate(references) if reference % 2]
			self.cache[word] = points
		return points


class Pyphen(object):
	def __init__(self, filename=None, lang=None, left=2, right=2, cache=True):
		if not filename:
			filename = LANGUAGES[language_fallback(lang)]
		self.left = left
		self.right = right
		if not cache or filename not in hdcache:
			hdcache[filename] = HyphDict(filename)
		self.hd = hdcache[filename]

	def positions(self, word):
		right = len(word) - self.right
		return [i for i in self.hd.positions(word) if self.left <= i <= right]

	def iterate(self, word):
		for position in reversed(self.positions(word)):
			if position.data:
				# get the nonstandard hyphenation data
				change, index, cut = position.data
				index += position
				if word.isupper():
					change = change.upper()
				c1, c2 = change.split('=')
				yield word[:index] + c1, c2 + word[index + cut:]
			else:
				yield word[:position], word[position:]

	def wrap(self, word, width, hyphen='-'):
		width -= len(hyphen)
		for w1, w2 in self.iterate(word):
			if len(w1) <= width:
				return w1 + hyphen, w2
	
	def fixer(self, data): #[wip]
		isV = lambda x:x in "aeiuo"
		data = list(data)
		#[wip] 
		if len(data) >= 3:
			if (data[0] == data[1]) or (isV(data[0]) != isV(data[1]) and isV(data[2])): # VX-VX =or= CX-CX or # A-VA-..
				data[0] += "-"
			else:
				pass
		
		
		return "".join(data)
	
	def inserted(self, word, hyphen='-'):
		word_list = list(word)
		for position in reversed(self.positions(word)):
			if position.data:
				# get the nonstandard hyphenation data
				change, index, cut = position.data
				index += position
				if word.isupper():
					change = change.upper()
				word_list[index:index + cut] = change.replace('=', hyphen)
			else:
				word_list.insert(position, hyphen)

		#return self.fixer(''.join(word_list))
		return ''.join(word_list)

	__call__ = iterate
