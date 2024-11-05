# Python assignment on Lists, Tuples, Sets and Dictionaries:
#
# Write a program consisting of multiple functions that does the following: (each function to perform a particular task)
#
# 1. Reads the text in a file in its same directory, given the file name
# Use a function that contains the following logic between the dotted lines
# -------------------------------------------
# # Read the text
# with open('text.txt', 'r') as file:
#     text = file.read()
# -------------------------------------------
#
# 2. Strips off the punctuation marks from the text
# . ? ! , ; : ( ) [ ] { } "
# Don't worry about dashes and single quote marks
# - '
# Use a function that contains the following logic between the dotted lines
# -------------------------------------------
# # Strip the punctuation
# text = text.replace('.', '')
# text = text.replace('?', '')
# text = text.replace('!', '')
# text = text.replace(',', '')
# text = text.replace(';', '')
# text = text.replace(':', '')
# text = text.replace('(', '')
# text = text.replace(')', '')
# text = text.replace('[', '')
# text = text.replace(']', '')
# text = text.replace('{', '')
# text = text.replace('}', '')
# text = text.replace('"', '')
# -------------------------------------------
#
# 3. Removes all types of blank spaces other than single blank space ' ' from the text
# Use a function that uses the string method split() to remove all '\t', '\n' an multiple blank spaces like '  '
#
# 4. converts each words to lower case
# Use a function that contains the following logic between the dotted lines
# -------------------------------------------
# # Make everything lower-case
# text = text.lower()
# -------------------------------------------
#
# 5. list out each unique word with its frequency.
#
# Example: If the text file contents are between the dotted lines
# -----------------------
# Hello,  world,  hello!
# World: hello?
#
# GOODBYE.
# -----------------------
#
# The output should be: (because "hello" occurs three times, "world" occurs twice, and "goodbye" occurs once)
# -----------------------
# hello: 3
# world: 2
# goodbye: 1
# -----------------------
#
# 6. Write another function to display the words and their frequencies to appear in alphabetical order of the words.
# Example
# -----------------------
# goodbye: 1
# hello: 3
# world: 2
# -----------------------
#
# 7. Write another function to display the words and their frequencies to appear in the order of
# frequency, from the most frequent down to the least frequent.
# Example
# -----------------------
# hello: 3
# world: 2
# goodbye: 1
# -----------------------
#
# 8. Modify the last function to display the words and their frequencies to appear in the descending
# order of frequency. The words with same frequency needs to be further displayed in alphabetical order.
#
# The program must have:
# - Correct logic
# - Free from syntax and runtime errors
# - Good choice of variable names
# - Sensible use of comments


def read_text_file(filename):
    """Reads the text from a file.
    Args:
        filename: The name of the file to read.
    Returns:
        The text from the file.
    """
    # Read the text
    with open(filename, 'r') as file:
        text = file.read()
    return text


def strip_punctuation(text):
    """Strips punctuation marks from the text.
    Args:
        text: The text to strip punctuation from.
    Returns:
        The text with punctuation removed.
    """
    # Strip the punctuation
    punctuation_marks = ['.', '?', '!', ',', ';', ':', '(', ')', '[', ']', '{', '}', '"']
    for mark in punctuation_marks:
        text = text.replace(mark, '')
    return text


def remove_extra_spaces(text):
    """Removes all types of blank spaces other than single blank space ' ' from the text.
    Args:
        text: The text to remove extra spaces from.
    Returns:
        The text with only single spaces.
    """
    # Remove extra spaces
    words = text.split()
    text = ' '.join(words)
    return text


def make_lowercase(text):
    """Converts each word to lowercase.
    Args:
        text: The text to convert to lowercase.
    Returns:
        The text in lowercase.
    """
    # Make everything lower-case
    text = text.lower()
    return text


def count_word_frequencies(text):
    """Counts the frequency of each unique word in the text.
    Args:
        text: The text to count word frequencies in.
    Returns:
        A dictionary where keys are unique words and values are their frequencies.
    """
    words = text.split()
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return word_counts


def display_words_alphabetically(word_counts):
    """Displays the words and their frequencies in alphabetical order."""
    # Sort the words alphabetically
    sorted_words = sorted(word_counts)
    # Print the words and their frequencies in alphabetical order
    for word in sorted_words:
        print(f"{word}: {word_counts[word]}")


def display_words_by_frequency(word_counts):
    """Displays the words and their frequencies in descending order of frequency."""
    # Sort the words by frequency in descending order
    sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    # Print the words and their frequencies in descending order of frequency
    for word, count in sorted_words:
        print(f"{word}: {count}")


def display_words_by_frequency_with_alphabetical_tiebreaker(word_counts):
    """Displays the words and their frequencies in descending order of frequency, with alphabetical tiebreaker."""
    # Sort the words by frequency in descending order, with alphabetical tiebreaker
    sorted_words = sorted(word_counts.items(), key=lambda item: (-item[1], item[0]))
    # Print the words and their frequencies in descending order of frequency, with alphabetical tiebreaker
    for word, count in sorted_words:
        print(f"{word}: {count}")


def main():
    """Main function to process text file and output word frequencies."""
    filename = 'text.txt'
    text = read_text_file(filename)
    text = strip_punctuation(text)
    text = remove_extra_spaces(text)
    text = make_lowercase(text)
    word_counts = count_word_frequencies(text)
    # Print the word frequencies
    print(f"Words in {filename} and their frequencies:")
    for word, count in word_counts.items():
        print(f"{word}: {count}")

    # Display the words and their frequencies in alphabetical order
    print("Words in alphabetical order:")
    display_words_alphabetically(word_counts)

    # Display the words and their frequencies in descending order of frequency
    print("\nWords in order of descending frequency:")
    display_words_by_frequency(word_counts)

    # Display the words and their frequencies in descending order of frequency, with alphabetical tiebreaker
    print("\nWords in order of descending frequency, with alphabetical tiebreaker:")
    display_words_by_frequency_with_alphabetical_tiebreaker(word_counts)


if __name__ == "__main__":
    main()
