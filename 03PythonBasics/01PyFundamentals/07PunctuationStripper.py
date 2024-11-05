# Read the text
with open('text.txt', 'r') as file:
    text = file.read()

# Strip the punctuation
text = text.replace('.', '')
text = text.replace(',', '')
text = text.replace('"', '')
text = text.replace("'", '')
text = text.replace(':', '')
text = text.replace(';', '')
text = text.replace('?', '')
text = text.replace('\n', ' ')
text = text.replace('  ', ' ')

# Make everything lower-case
text = text.lower()

# Save the results
with open('result.txt', 'w') as file:
    file.write(text)