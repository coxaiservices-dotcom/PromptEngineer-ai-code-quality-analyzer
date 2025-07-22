def calculate_sum(numbers):
    """This function calculates the sum of a list of numbers."""
    # Initialize the result variable
    result = 0
    
    # Iterate through each number in the list
    for number in numbers:
        # Add the current number to the result
        result += number
    
    # Return the final result
    return result

def main():
    # Create a sample list of numbers
    sample_data = [1, 2, 3, 4, 5]
    
    # Calculate the sum
    total = calculate_sum(sample_data)
    
    # Print the result
    print(f"The sum is: {total}")

if __name__ == "__main__":
    main()