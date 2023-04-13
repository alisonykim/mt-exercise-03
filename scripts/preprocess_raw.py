import sys
import re


with open(0, 'rb') as f_in:
	for line in f_in:
		line = line.decode('utf-8', errors='ignore').strip()
		html_tags = r'\<.+\>'
		parentheticals = r'\(.+\)'
		css_tags = r'\#+.+'
		digits = r'\d+'
		salutations = r'(The|Dr|Mr|Mrs|Ms)\.+([A-Z]){2,}'
		remove = '|'.join([html_tags, parentheticals, css_tags, digits, salutations])
		line = re.sub(remove, '', line)
		if line.upper() != line and len(line.split()) > 5: # exclude character names and short lines
			sys.stdout.write(line + '\n')