# Q-01 # How many syntax errors does the following code contain?
# print ("Hi'   }

# Q-02 # If you enter 6 when prompted by the following code it causes an error. Why?
# num = input('Please enter a number: ')
# print(num + 2)

# Q-03 # Which lines contain an indentation syntax error?
# if (25 > 10):
#             print('25 is big')
#     else:
#                     print('25 is small')

# Q-04 # How to comment-out multiple lines of code, such as the following three:
# x = 10
# x = x + 20
# print(x)

# Q-05 # Which one of the following statements about Python objects and their attributes is true?
# A. Attributes can be functions but they need not be
# B. Attributes can only be functions
# C. Attributes cannot be functions
# D. None of the above are true
# A is correct: Attributes can be functions but they need not be.
# Attributes in Python are variables associated with an object. They define the object's state and behavior.
# Python allows to define functions as attributes of an object.

# Q-06 # In Python, how to apply the function func to the word "Hello"?
# A. func(Hello)
# B. func('Hello')
# C. 'Hello'(func)
# D. func.Hello()
# B is correct:

# Q-07 # Which one of the following is not a literal?
# A. X
# B. "X"
# C. 2
# D. "2"
# The answer is A. X. Literals in programming represent fixed values directly written into the code.
# They don't need to be calculated or derived from anything else.

# Q-08 # Which one of the following claims about variables in Python is true?
# A. Variables must be declared before they can be assigned a value
# B. Once assigned a value, a variable cannnot be assigned a different value of the same type
# C. Once assigned a value, a variable cannnot be assigned a different value of a different type
# D. None of the above are true
# The answer is D.

# Q-09 # What should you infer from the following line of code:
# FACTOR = 1.453
# A. FACTOR is an important variable
# B. FACTOR is a constant and trying to change its value will cause an error
# C. The value of FACTOR is intended to not change
# D. FACTOR can be used as a literal for the number 1.453
# The correct answer is C. The value of FACTOR is intended to not change.

# Q-10 # Some expressions are simple and some are complex. Which one of the following Python expressions is a complex expression?
# A. 23.5e-6
# B. "Hello, world!"
# C. my_long_variable_name
# D. 1 + 2
# D. 1 + 2: This is a complex expression because it involves two operands (1 and 2) and an operator (+). It is evaluated to produce the result 3.

# Q-11 # Consider the following odd-looking line of code in Python:
# x(x)
# Are there any things we could substitute x with such that it will not cause an error? Yes/No?
# YES

# Q-12 # Consider the following two statements:
# print(2 * '42')
# print(2 + '42')
# Why does the second statement cause an error whereas the first does not?
# A. You can add integers in Python but you cannot multiply them.
# B. You can add strings in Python but you cannot multiply them.
# C. You can multiply a string by an integer in Python but you cannot add an integer to a string.
# D. You can add a string to an integer in Python but you cannot multiply a string by an integer.
# C

# Q - 13 # Why does the following Python code output "who" rather than "WHO"?
# INITIALS = 'who'
# INITIALS.upper()
# print(INITIALS)
# A. There is no upper method for strings in Python.
# B. INITIALS is a constant and cannot be changed.
# C. The print function undoes the effect of upper method.
# D. The upper method does not modify a string in place - it returns a new string.
# D

# Q-14 # Why doesn't the following code generate an error, even though cancelled is not defined?
# done = True
# if (done or cancelled):
#     print('Over')
# A. Python implicitly defines cancelled for us.
# B. cancelled is a Python keyword and does not need to be defined.
# C. or is a short-circuiting operator, so cancelled is not evaluated.
# D. if is a short-circuiting operator, so cancelled is not evaluated.
# C

# Q-15 # Why does the following piece of code generate an error?
# x = 5
# if (x = 5):
#     print('x is five')
# Ans: x==5

# Q-16 # Consider the following piece of Python code, containing a while statement:
# How many times does line 4 get executed?
# x = 1
# while (x <= 10):
#     if (x == 4): break
#     x = x + 1
# Ans: 3 times

# Q-17 # This Python code generates an error if the user enters a letter:
# value = input('Please enter an integer: ')
# value = int(value)
# value = value**2
# print(value)
# You can prevent the error by using a try ... except ... statement. In which one of the following ways could you use it?
# Note that you may need to arrange the code properly in addition to wrapping that line of code to handle that exception.
# A. Wrap it around line 1
# B. Wrap it around line 2
# C. Wrap it around line 3
# D. Wrap it around line 4
# The correct answer is B. Wrap it around line 2.

# Q-18 # Some statements are simple and some are compound. What is the defining difference?
# A. Simple statements are easy to understand whereas compound statements are difficult.
# B. Simple statements are fast o execute whereas compound statements are slow.
# C. Simple statements contain other statements whereas compound statements do not.
# D. Compound statements contain other statements whereas simple statements do not.
# D.

# Q-19 # In which one of the following ways could you use the ceil function of the math module?
# import math
# x = math.ceil(3.5)
# import math as m
# x = m.ceil(3.5)
# from math import ceil
# x = ceil(3.5)
# print(f'x = {x}')

# Q-20 # Suppose you want to open a file called "results.txt" and write some results to it, but only if the file does not already exist. Which one of the following statements should you use?
# A. f = open('results.txt', 'r')
# B. f = open('results.txt', 'w')
# C. f = open('results.txt', 'x')
# D. f = open('results.txt', 'a')
# Correct answer: C. f = open('results.txt', 'x')
