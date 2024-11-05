def get_integer_input():
    """
    This function takes console input until a valid integer is entered.
    Returns:
        int: The entered integer value.
    """
    while True:
        try:
            value = int(input("Please enter an integer: "))
            return value  # Return the integer if conversion is successful
        except ValueError:
            print("Invalid input. Please enter an integer.")


def even_or_odd(number):
    """
    Determines if a given integer is even or odd.
    Args:
        number: The integer to check.
    Returns:
        str: "Even" if the number is even, "Odd" otherwise.
    """
    if number % 2 == 0:
        return "Even"
    else:
        return "Odd"


# Ask the user for an integer
integer = get_integer_input()
# Determine if it is even/ odd
result = even_or_odd(integer)

# Show the result
print(f'You entered {integer}. It is \"{result}\".')
