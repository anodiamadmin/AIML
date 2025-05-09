class Temperature:
    """
    Represents a temperature value with unit conversion and comparison capabilities.
    """

    def __init__(self, value, unit):
        """
        Initializes a Temperature object.
        Args:
            value: The numerical value of the temperature.
            unit: The unit of the temperature (C, F, or K).
        """
        self.value = value
        self.unit = unit.upper()

    def __str__(self):
        """
        Returns a string representation of the Temperature object.
        """
        return f"Temperature: {self.value}{self.unit}"

    def convert_to(self, new_unit, decimal_places=None):
        """
        Converts the temperature to the specified unit.
        Args:
            new_unit: The target unit (C, F, or K).
            decimal_places: Optional number of decimal places to round the result to.
        Returns:
            The converted temperature value, optionally rounded.
        """
        new_unit = new_unit.upper()
        if self.unit == new_unit:
            return self.value
        # Convert Celsius to Fahrenheit & Kelvin
        elif self.unit == 'C':
            if new_unit == 'F':  # Celsius to Fahrenheit
                return round((9 / 5 * self.value) + 32, decimal_places) if decimal_places else (9 / 5 * self.value) + 32
            elif new_unit == 'K':  # Celsius to Kelvin
                return round(self.value + 273.15, decimal_places) if decimal_places else self.value + 273.15
        # Convert Fahrenheit to Celsius & Kelvin
        elif self.unit == 'F':
            if new_unit == 'C':  # Fahrenheit to Celsius
                return round((self.value - 32) * 5 / 9, decimal_places) if decimal_places else (self.value - 32) * 5 / 9
            elif new_unit == 'K':  # Fahrenheit to Kelvin
                return round(((self.value - 32) * 5 / 9) + 273.15, decimal_places) if decimal_places else ((
                               self.value - 32) * 5 / 9) + 273.15
        # Convert Kelvin to Celsius & Fahrenheit
        elif self.unit == 'K':
            if new_unit == 'C':  # Kelvin to Celsius
                return round(self.value - 273.15, decimal_places) if decimal_places else self.value - 273.15
            elif new_unit == 'F':  # Kelvin to Fahrenheit
                return round(((self.value - 273.15) * 9 / 5) + 32, decimal_places) if decimal_places else ((
                               self.value - 273.15) * 9 / 5) + 32
        else:
            raise ValueError("Invalid unit of temperature! Temperature should be in C, F or K")

    def __eq__(self, other):
        """
        Compares two Temperature objects for equality.
        Args:
            other: The other Temperature object to compare against.
        Returns:
            True if the temperatures are equal in the same unit, False otherwise.
        """
        if self.unit == other.unit:
            return self.value == other.value
        else:
            return self.convert_to(other.unit) == other.value

    def __lt__(self, other):
        """
        Compares two Temperature objects for less than.
        Args:
            other: The other Temperature object to compare against.
        Returns:
            True if the first temperature is less than the second in the same unit, False otherwise.
        """
        if self.unit == other.unit:
            return self.value < other.value
        else:
            return self.convert_to(other.unit) < other.value

    def __le__(self, other):
        """
        Compares two Temperature objects for less than or equal to.
        Args:
          other: The other Temperature object to compare against.
        Returns:
          True if the first temperature is less than or equal to the second in the same unit, False otherwise.
        """
        if self.unit == other.unit:
            return self.value <= other.value
        else:
            return self.convert_to(other.unit) <= other.value

    def __gt__(self, other):
        """
        Compares two Temperature objects for greater than.
        Args:
          other: The other Temperature object to compare against.
        Returns:
          True if the first temperature is greater than the second in the same unit, False otherwise.
        """
        if self.unit == other.unit:
            return self.value > other.value
        else:
            return self.convert_to(other.unit) > other.value

    def __ge__(self, other):
        """
        Compares two Temperature objects for greater than or equal to.
        Args:
          other: The other Temperature object to compare against.
        Returns:
          True if the first temperature is greater than or equal to the second in the same unit, False otherwise.
        """
        if self.unit == other.unit:
            return self.value >= other.value
        else:
            return self.convert_to(other.unit) >= other.value
