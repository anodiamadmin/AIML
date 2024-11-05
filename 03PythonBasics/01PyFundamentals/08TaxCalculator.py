def calc_tax(income):
    """
    This function calculates tax for a given income.
     - param: income
     - return: tax
    """
    if income <= 18200:
        return 0
    elif income <= 45000:
        return round((income - 18200) * 0.19)
    elif income <= 120000:
        return round(5092 + (income - 45000) * 0.325)
    elif income <= 180000:
        return round(29467 + (income - 120000) * 0.37)
    else:
        return round(51667 + (income - 180000) * 0.45)


# Input income: Assuming user enters a valid +ve integer for one time only.
income_22_23 = int(input("What was your income in 2022-23? "))

# Calculate Tax for the entered value of income
tax = calc_tax(income_22_23)

# Print the estimated tax on income in $, round off to the nearest integer
print("The estimated tax on your income is $", tax)
