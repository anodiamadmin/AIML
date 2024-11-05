def get_height_input():
    """
    This function takes console inputs until valid numerical value for height is entered.
    Returns:
        float: height.
    """
    while True:
        try:
            height = float(input("What is your height in meters? "))
            if height <= 0:
                print("Height must be a positive number. Please try again.")
                continue
            elif height < 0.1:
                print(f"Height entered ({height}m) is too short. Please enter a valid human height between 0.1m and 4m.")
                continue
            elif height > 4:
                print(f"Height entered ({height}m) is too tall. Please enter a valid human height between 0.1m and 4m.")
                continue
            return height
        except ValueError:
            print("Invalid height. Please enter the valid numerical value for your height in meters.")


def get_weight_input():
    """
    This function takes console inputs until valid numerical value for weight is entered.
    Returns:
        float: weight.
    """
    while True:
        try:
            weight = float(input("What is your weight in kilograms? "))
            if weight <= 0:
                print("Weight must be a positive number. Please try again.")
                continue
            elif weight < 0.5:
                print(f"Weight entered ({weight}Kg) is too low. Please enter a valid human weight between 0.5 and 1000Kgs.")
                continue
            elif weight > 1000:
                print(f"Weight entered ({weight}Kg) is too high. Please enter a valid human weight between 0.5 and 1000Kgs.")
                continue
            return weight
        except ValueError:
            print("Invalid weight. Please enter the valid numerical value for your weight in Kgs.")


def calculate_bmi(height, weight):
    """
    This function calculates BMI.
    Args:
        height: height.
        weight: weight.
    Returns:
        float: BMI.
    """
    bmi = weight / (height ** 2)
    return bmi


# Ask the user for height
height = get_height_input()

# Ask the user for weight
weight = get_weight_input()

# Calculate BMI
bmi = calculate_bmi(height, weight)

# Display BMI with 2 decimal places
print(f'Your  BMI is {bmi:.2f}.')
