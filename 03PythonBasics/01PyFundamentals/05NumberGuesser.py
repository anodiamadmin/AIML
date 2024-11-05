# Ask the user to pick a number
print('Please pick a number between 1 and 100 (inclusive)')

# Initialise variables
lower_bound = 1
upper_bound = 100
got_it = False

# Keep guessing until correct
while got_it == False:

    # Guess the number in the middle of the lower and upper bounds
    guess = round((lower_bound + upper_bound) / 2)

    # Display the guess
    print(f'My guess is: {guess}')

    # Ask the user for feedback
    result = input('Is my guess too low (l), too high (h), or correct (c)? ')

    # Adjust accordingly
    if result == 'l':
        lower_bound = guess + 1
    elif result == 'h':
        upper_bound = guess - 1
    elif result == 'c':
        got_it = True

# Finish
print('I got it!')