from temperature import Temperature

# Create Temperature objects
temp_1 = Temperature(32, 'C')
temp_2 = Temperature(100, 'F')
temp_3 = Temperature(324, 'K')

# Print them
print(temp_1)  # Outputs Temperature: 32C
print(temp_2)  # Outputs Temperature: 100F
print(temp_3)  # Outputs Temperature: 324K

# Convert them
print(temp_1.convert_to('F'))  # Outputs 89.6
print(temp_2.convert_to('K', 3))  # Outputs 310.928
print(temp_3.convert_to('C', 1))  # Outputs 50.9

# Compare them
print(temp_1 == temp_2)  # Outputs False
print(temp_1 < temp_2)  # Outputs True
print(temp_1 <= temp_2)  # Outputs True
print(temp_1 > temp_2)  # Outputs False
print(temp_1 >= temp_2)  # Outputs False
