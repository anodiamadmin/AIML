# ANS: 1.C, 2.B, 3.D, 4.B, 5.D, 6.C, 7.B, 8.D, 9.D, 10.D,
# 11.D, 12.D, 13.D, 14.C, 15.C, 16.D , 17.D , 18.C , 19.D , 20.B

# Q01: What is the difference between a list and a tuple?
# A. A list is ordered but a tuple is not.
# B. A list can contain elements of different types but a tuple cannot.
# C. A list is mutable but a tuple is not.
# D. A list can contain duplicate elements but a tuple cannot.
# Ans: C
#
# Q02: An ordinary playing card has two attributes: a rank ('A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K'),
# and a suit ('Hearts', 'Spades', 'Diamonds', 'Clubs'). When considering all 52 playing cards, which of the
# following collections would be the most appropriate for representing a single playing card?
# A. A list, e.g. ['J', 'Spades']
# B. A tuple, e.g. ('J', 'Spades')
# C. A set, e.g. {'J', 'Spades'}
# D. A dictionary, e.g. {'J': 'Spades'}
# Ans: B

# Q03: Suppose you're writing a program to work with the grades of students in a class. Each student has a
# unique id, but different students might get the same grade. Which one of the following collections would
# best allow you to store and modify this data?
# A list of tuples of the form [(2345, 'CR'), (4567 , 'HD'), …]
# A tuple of sets of the form ({2345, 'CR'}, {4567 , 'HD'}, …)
# A tuple of sets of the form ({2345, 4567, …}, {'CR', 'HD', …})
# A dictionary of the form {2345: 'CR', 4567: 'HD', …}
# Ans: D

# Q04: Which one of the following expressions is a list literal?
# A. list (1, 3, 9)
# B. [x ** 0, x ** 1, x ** 2]
# C. [3 ** 0, 3 ** 1, 3 ** 2]
# D. None of them are
# Ans: B

# Q05: The following code was written to show the intersection of the empty set with another set, but it
# generates an error. Why?
# s = {'a', 'b', 'c'}
# print({}.intersection(s))
# A. Python does not support empty sets.
# B. Sets do not have an intersection method.
# C. The intersection method of sets cannot be used with the empty set.
# D. {} is not the empty set - it's the empty dictionary.
# Ans: D

# Q06: Which one of the following statements is true?
# A. You can use the len function to find the number of elements in a list, tuple, set, or dictionary.
# B. You can use the len function to find the number of elements in a list or tuple, but not in a set or
# dictionary because they are not ordered.
# C. You can use the len function to find the number of elements in a list, tuple or set, but not in a
# dictionary because their elements are key-value pairs.
# D. You can use the len function to find the number of elements in a list, tuple, or dictionary, but not
# in a set because their elements must be unique.
# Ans: C

# Q07: Which one of the following statements could you use to extend a list a by appending the elements of a list b?
# For instance, if a = [1, 2, 3] and b=['a', 'b', 'c']
# The expected result should look like a = [1, 2, 3, 'a', 'b', 'c']
# 	A. a. append (b)
# 	B. a. extend (b)
# 	C. a+b
# 	D. You could use any of the above.
# Ans: B

# Q08: Which one of the following statements could you use to add the letter 'a' as an item to the end
# of a tuple t?
# A. t.add('a')
# B. t.append('a')
# C. t = t + 'a'
# D. None of the above
# Ans: D

# Q09: Suppose that letters is a list of ten letters. Which one of the following pieces of code could
# you use to change the first two elements to 'a' and 'b'?
# A. letters[0:1]=['a', 'b']
# B. letters[0:1]=('a', 'b')
# C. letters[0]='a'
#    letters[1]='b'
# D. Any of the above
# Ans: D

# Q10: Why does the following piece of code print b?
# x = ['b', 'b']
# print(x.pop())
# A. The pop method removes the first element and returns the remaining second element.
# B. The pop method removes the second element and returns the remaining first element.
# C. The pop method removes and returns the first element.
# D. The pop method removes and returns the second element.
# Ans: D

# Q11: Which one of the following is true?
# A. The sort() method, when called on a list, and the sorted() function, when applied to a list, both
# return the same thing - the sorted list.
# B. The sorted() function cannot be applied to tuples because tuples are immutable.
# C. Sets have a sort() method but there is little point in using it because sets are not ordered.
# D. None of the above are true.
# Ans: D

# Q12: Suppose you have a list called "names" which contains a number of names. Suppose you want to join
# them into a string, separated by commas. Which one of the following expressions could you use?
# A. names. Join(", ")
# B. join(names, ",")
# C. join(",", names)
# D. ", ". join(names)
# Ans: D

# Q13: Suppose you want to loop through a dictionary d and print each of its values. Which one of the
# following pieces of code could you use?
# A.
# for x in d:
#     print(d[x])
# B.
# for x, y in d.items():
#     print(y)
# C.
# for x in d.values():
#     print(x)
# D.You could use any of the above.
# Ans: D

# Q14: Which one of the following is true?
# A> Lists can contain lists but tuples cannot contain lists.
# B> Sets can contain lists, tuple, or dictionaries but they cannot contain sets.
# C> Dictionaries can have sets as their values but not as their keys.
# D> None of the above are true.
# Ans: C

# Q15: In the code below, line 2 will produce an error but line 3 will not. Why?
# x = ([1], [2], [3])
# x[0] = 4 # Error
# x[0][0] = 4 # No error
# A> x[0] does not exist, but x[0][0] does.
# B> You can't assign a number to because the other items in the tuple are lists. But you can assign a
# number to x[0][0]
# C> You can't change which items are in the tuple, so you can't replace the first item by 4. But you can change
# an item internally, if it is mutable. So you can replace the first element of the first item by 4.
# D> None of the above is correct.
# Ans: C

# Q16: Suppose that 'word' is a variable that refers to a word. Suppose you want an expression that
# returns the set of consonants in word. Which one of the following expressions could you use?
# A. {x for x in word if x not in {'a', 'e', 'i', 'o', 'u'}}
# B. {x for x in word if not x in ('a', 'e', 'i', 'o', 'u')}
# C. {x for x in word}.difference ({'a', 'e', 'i', 'o', 'u'})
# D. You could use any of the above.
# Ans: D

# Q17: Consider the following code template:
# for i in <expression>:
#     <statements>
# Suppose you want i to loop through the numbers 0, 2, 4, 6, 8 and 10. Which one of the following could
# you use to replace <expression>?
# A> (0, 2, 4, 6, 8, 10)
# B> range(0, 11, 2)
# C> [2*x for x in range(0, 6)]
# D> You could use any of the above.
# Ans: D

# Q18: Why does the following piece of code generate an error?
# word = 'Expediensy'
# word[-2] = 'c'
# A> word is a string and the indexing operator [] cannot be used with strings.
# B> When using the indexing operator [ ] with strings, only positive indices can be used.
# C> The indexing operator [ ] can be used to select string elements but not to modify them.
# D> None of the above are true.
# Ans: C

# Q19: Suppose the following piece of code executes without error:
# with open('myfile', 'r') as file:
#     lines = file.readlines()
# What will be the value of lines?
# A> A string containing the text in the file.
# B> A list of characters in the file.
# C> A list of lines in the file, not including their newline characters.
# D> A list of lines in the file, including their newline characters.
# Ans: D

# Q20: Suppose you're creating a datetime object from the string "June 24, 1968, at 05:30" using
# datetime.strptime. Which one of the following format strings should you use?
# A> '%b %d, %y, at %H:%M'
# B> '%B %d, %Y, at %H:%M'
# C> '%B %d, %y, at %H:%M'
# D> '%B %d, %Y, at %h:%m'
# Ans: B
