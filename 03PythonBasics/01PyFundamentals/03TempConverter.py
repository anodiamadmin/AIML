# Ask the user for a temperature:
input_temp = input('Enter a temperature (e.g. 24C, 78F): ')

# Clean the input by making it all uppercase and removing any spaces:
input_temp = input_temp.upper().replace(' ', '')

# Break it into a number and a scale:
input_number = float(input_temp[:-1])
input_scale = input_temp[-1]

# Convert it:
if input_scale == 'C':
    output_number = round((input_number * 1.8) + 32, 1)
    output_scale = 'F'
elif input_scale == 'F':
    output_number = round((input_number - 32) / 1.8, 1)
    output_scale = 'C'

# Another alternative for lines 7-17 without using slices which will be introduced in Week 2:
#if input_temp.endswith('C'):
#    input_number = float(input_temp.replace('C', '')) # Remove the scale and convert to float
#    output_number = round((input_number * 1.8) + 32, 1)
#    output_scale = 'F'
#elif input_temp.endswith('F'):
#    input_number = float(input_temp.replace('F', '')) # Remove the scale and convert to float
#    output_number = round((input_number - 32) / 1.8, 1)
#    output_scale = 'C'

# Print the result:
print(f'{input_temp} is {output_number}{output_scale}')