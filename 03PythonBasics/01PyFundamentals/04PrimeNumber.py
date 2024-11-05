# Ask the user for a number to check
number_to_check = input('What number would you like to check? ')

# Convert the input to an integer
number_to_check = int(number_to_check)

if number_to_check < 2:
    print(f'{number_to_check} is not a prime')
else:
    # Check whether it is prime
    factor_to_try = 2
    result = 'prime'
    while factor_to_try < number_to_check:
        if number_to_check % factor_to_try == 0:
            result = 'not prime'
            break
        factor_to_try += 1

    # Show the result
    print(f'{number_to_check} is {result}')

    # Optional extra:
    if result == 'not prime':
        residual = int(number_to_check / factor_to_try)
        print(f'{number_to_check} = {factor_to_try} x {residual}')