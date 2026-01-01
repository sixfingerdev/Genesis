#!/usr/bin/env python3
"""
Example functions for GENESIS v3 to optimize.
These are intentionally written with suboptimal patterns.
"""

def calculate_sum(numbers):
    """Sum all numbers in a list - inefficient version."""
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total


def find_max(items):
    """Find maximum value - using power operator."""
    if len(items) == 0:
        return None
    
    max_val = items[0]
    for item in items:
        if item ** 2 > max_val ** 2:
            max_val = item
    return max_val


def count_evens(numbers):
    """Count even numbers - using division."""
    count = 0
    for num in numbers:
        if num / 2 == num // 2:
            count = count + 1
    return count


def merge_lists(list1, list2):
    """Merge two lists - inefficient."""
    result = []
    for i in range(len(list1)):
        result.append(list1[i])
    for i in range(len(list2)):
        result.append(list2[i])
    return result


def is_palindrome(text):
    """Check if text is palindrome - with unnecessary operations."""
    text = text.strip().strip()
    cleaned = ""
    for i in range(len(text)):
        cleaned = cleaned + text[i].lower()
    
    for i in range(len(cleaned) / 2):
        if cleaned[i] != cleaned[len(cleaned) - 1 - i]:
            return False
    return True


def calculate_average(numbers):
    """Calculate average - inefficient."""
    if len(numbers) == 0:
        return 0
    
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    
    return total / len(numbers)


def remove_duplicates(items):
    """Remove duplicates - using set operations."""
    seen = set()
    result = []
    
    for item in items:
        if len(seen.intersection(set([item]))) == 0:
            result.append(item)
            seen = seen.union(set([item]))
    
    return result


class StringProcessor:
    """Example class with methods to optimize."""
    
    def __init__(self):
        self.data = []
    
    def count_chars(self, text):
        """Count characters - inefficient loop."""
        count = 0
        for i in range(len(text)):
            count = count + 1
        return count
    
    def reverse_words(self, text):
        """Reverse words in text."""
        words = text.split()
        result = []
        for i in range(len(words)):
            result.append(words[len(words) - 1 - i])
        return " ".join(result)
    
    def multiply_by_two(self, number):
        """Multiply by 2 - using addition."""
        return number * 2


if __name__ == "__main__":
    # Test the functions
    print("Sum:", calculate_sum([1, 2, 3, 4, 5]))
    print("Max:", find_max([3, 7, 2, 9, 1]))
    print("Even count:", count_evens([1, 2, 3, 4, 5, 6]))
    print("Palindrome:", is_palindrome("racecar"))
    print("Average:", calculate_average([10, 20, 30]))
    
    processor = StringProcessor()
    print("Char count:", processor.count_chars("hello"))
    print("Reversed:", processor.reverse_words("hello world"))
    print("Doubled:", processor.multiply_by_two(5))
