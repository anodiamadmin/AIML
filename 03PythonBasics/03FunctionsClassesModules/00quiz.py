# Ans: 1.D, 2.D, 3.C, 4.D, 5.D, 6.D, 7.C, 8.A, 9.A,
# 10.D, 11.D, 12.D, 13.C, 14.D, 15.D, 16.A, 17.A, 18.C, 19.B, 20.D

# Q01: Suppose you are defining a function to check whether a certain mark is a passing mark. Which one of the following could you use as a name for the function?
# A> pass
# B> Pass
# C> $pass
# D> None of the above
# Ans: D

# Q02: Suppose you're defining a function called "go" but haven't yet figured out how to define it. Which one of the following placeholder definitions could you temporarily use?
# A> def go(): return
# B> def go(): pass
# C> def go(): {}
# D> Any of the above
# Ans: D

# Q03: Consider the following function definition:
# def repeat(s, n):
#     return s * n
# Which one of the following is true?
# A> s is an argument of repeat.
# B> s is an argument of a call to repeat.
# C> s is a parameter of repeat.
# D> s is a parameter of a call to repeat.
# Ans: C

# D04: Suppose a function is defined as follows:
# def f(x, y = 1):
#     ... more code ...
# Which one of the following is not a valid way to call the function?
# A> print(f(1))
# B> print(f(x=1))
# C> print(f(y=2, x=2))
# D> print(f(y=2))
# Ans: D

# Q05: Suppose a function is defined as follows:
# def f(x, y, *z):
#     ... more code ...
# Which one of the following is not a valid way to call the function?
# A> print(f(1, 2, 3, 4))
# B> print(f([1, 2], [3, 4]))
# c> print(f([1], [2], [3], [4]))
# D> print(f([1, 2, 3, 4]))
# Ans: D

# Q06: Which one of the following is true?
# a> A function definition must contain a return statement.
# b> A function definition cannot contain more than one return statement.
# c> A function always returns a value.
# d> A function can return more than one value.
# Ans: D

# Q07: Which one of the following functions has side effects?
# A>
# def cube(x):
#     return x**3
# B>
# def all_caps(x):
#     return x.upper()
# C>
# def grow(x):
#     return x.append('a')
# D>
# All of the above functions have side effects.
# Ans: C

# Q08: Consider the following program:
# def f(x):
#     def g(x):
#         x = 2
#         return x**2 + 3*x + 1
#     x = 5
#     return x*g(x)
# x = 10
# print(f(x))
# Just before line 4 executes, what is the value of the variable x that is defined in line 7?
# a> 2
# b> 5
# c> 10
# d> None
# Ans: A

# Q09: Consider the following piece of code:
# def g(x):
#     def f(x): return x**2 + 2*x + 1
#     return f(x)**2
# What kind of a function is f?
# A. Nested
# B. Higher-order
# C. Lambda
# D. Recursive
# Ans: A

# Q10: Consider the following piece of code:
# def f(x):
#     if x == 1: return 1
#     else: return x * f(x - 1)
# What kind of a function is f?
# A. Nested
# B. Higher-order
# C. Lambda
# D. Recursive
# Ans: D

# Q11:  Consider the following line of code:
# f = lambda x: x**3
# Which one of the following is true?
# A. f is a lambda function.
# B. f is assigned using a lambda function.
# C. f can be used as a lambda function.
# D. All of the above are true.
# Ans: D

# Q12: Which one of the following is true?
# A. Functions can have attributes.
# B. Functions can be supplied as arguments to functions.
# C. Functions can be returned by functions.
# D. All of the above are true.
# Ans: D

# Q13: The following program gives an error at line 4. Why?
# def vowels():
#     for x in ['a', 'e', 'i', 'o', 'u']:
#         yield(x)
# print(vowels()[4])
# A> There is no 4th item in the list that vowels returns.
# B> You have to first assign the list that vowels returns to a variable, and then you can access the elements of the list using square brackets with that variable.
# C> vowels does not return a list.
# D> None of the above are true.
# Ans: C

# Q14: Which one of the following is true?
# A> Classes can have instances which can have attributes which can be functions.
# B> Classes can have attributes too, which can also be functions.
# C> Instance attributes that are functions are called instance methods; class attributes that are functions are called class methods.
# D> All of the above are true.
# Ans: D

# Q15: The following program defines a class Triangle and attempts to create an instance of it, but there is an error in line 4. Why?
# class Triangle:
#     def __init__(sides):
#         self.sides = sides
# t = Triangle([3, 4, 5])
# The parameter sides cannot be assigned a list of values.
# The parameter sides must be given a default value of self.
# The function __init__ must have a self parameter after sides.
# The function __init__ must have a self parameter before sides.
# Ans: D

# Q16: The following piece of code produces an error at line 8. Why?
# class Name:
#     def __init__(self, x, y):
#         self.first_name = x
#         self.last_name = y
#     def initials():
#         return (self.first_name[0] + self.last_name[0]).upper()
# my_name = Name('John', 'Smith')
# print(my_name.initials())
# A> initials is defined as a class method but is being used as an instance method.
# B> initials is defined as an instance method but is being used as a class method.
# C> initials is defined as a class method but classes cannot have methods.
# D> initials is defined as an instance method but instances cannot have methods.
# Ans: A

# Q17: Suppose you are defining a Rectangle class, and have given each instance a width
# attribute and a length attribute. Which one of the following would be the most sensible
# definition of an __eq__ special method?
# A>
# def __eq__(self, other):
#     return (self.width == other.width) and (self.length == other.length)
# B>
# def __eq__(self, other):
#     return (self.width == other.width) or (self.length == other.length)
# C>
# def __eq__(self, other):
#     return (2*(self.width + self.length)) == (2*(other.width + other.length))
# D>
# def __eq__(self, other):
#     return (self.width * self.length) == (other.width * other.length)
# Ans: A

# Q18: In which one of the following pairs of class definitions does class B inherit from class A?
# A>
# class A:
#     val = 1
# class B:
#     val = A
# B>
# class A:
#     val = 1
# class B:
#     val = A()
# C>
# class A:
#     val = 1
# class B(A):
#     val = 1
# D>
# In none of the above does class B inherit from class A.
# Ans: C

# Q19: Consider the following piece of code:
# class Person:
#     def __init__(self, name, boss = None):
#         self.name = name
#         self.boss = boss
# a = Person('Anna', Person('Bree'))
# ... more code ...
# Just after line 5 executes, which one of the following is true?
# A. a has an attribute called "boss" whose value is the string "Bree".
# B. a has an attribute called "boss" whose value is an instance of Person which has an attribute called "boss"
# whose value is None.
# C. a has an attribute called "boss" whose value is an instance of Person but has no "boss" attribute.
# D. None of the above are true.
# Ans: B

# Q20: Suppose you've created a module called "my_functions.py". Suppose it contains a
# function called "trigger". Suppose you want to apply this function to a variable called
# "x" in your current program, using the following statement:
# print(trigger(x))
# Which of the following import statements should you have in your program?
# A. import my_functions
# B. import my_functions as trigger
# C. import trigger from my_functions
# D. from my_functions import trigger
# Ans: D
