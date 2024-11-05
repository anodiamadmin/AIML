# Ask the user how many Fibonaccis she'd like
num_to_print = input('How many Fibonacci numbers would you like? ')

# Convert the input to an integer
num_to_print = int(num_to_print)

# Print the requested number of Fibonaccis
if num_to_print >= 1:
    print(0)
    two_back = 0
if num_to_print >= 2:
    print(1)
    one_back = 1
if num_to_print >= 3:
    counter = 3
    while counter <= num_to_print:
        current = two_back + one_back
        print(current)
        two_back = one_back
        one_back = current
        counter += 1