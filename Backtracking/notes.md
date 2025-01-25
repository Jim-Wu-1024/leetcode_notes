### 1286. Iterator for Combination

```python
from typing import List

class CombinationIterator:

    def __init__(self, characters: str, combinationLength: int):
        self.string = characters
        self.length = combinationLength
        self.combinations = []  # List to store all generated combinations
        self.__backtracing(0, [])  # Generate combinations using backtracking
        self.index = 0  # Pointer to the current combination in the list

    def __backtracking(self, start: int, path: List[str]) -> None:
        # Base case: If the current combination has the required length, store it
        if len(path) == self.length:
            self.combinations.append(('').join(path[:]))  # Convert list to string and append
            return

        # Iterate through the string to generate combinations
        # Use pruning to ensure we don't exceed the required length
        for i in range(start, len(self.string) - self.length + len(path) + 1):
            path.append(self.string[i])  # Add the current character to the combination
            self.__backtracing(i + 1, path)  # Recurse to generate further combinations
            path.pop()  # Backtrack by removing the last character

    def next(self) -> str:
        combination = self.combinations[self.index]  # Retrieve the combination at the current index
        self.index += 1  # Move to the next combination
        return combination

    def hasNext(self) -> bool:
        return self.index < len(self.combinations)

```

### 2397. Maximum Rows Covered by Columns

```python
class Solution:
    def maximumRows(self, matrix: List[List[int]], numSelect: int) -> int:
        # Helper function to calculate the number of rows fully covered by selected columns
        def ifCovered(selected: List[int]) -> int:
            num = 0
            for row in range(m):
                # Compute the uncovered value for the current row
                value = row_value[row]
                for col in selected:
                    value -= matrix[row][col]  # Subtract the value in the selected columns
                # If value becomes 0, the row is fully covered
                num += 1 if value == 0 else 0
            return num

        # Backtracking function to try all column combinations
        def backtrack(start: int, selected: List[int]) -> None:
            # Base case: when we've selected the required number of columns
            if len(selected) == numSelect:
                # Update the maximum covered rows
                self.max_covered = max(self.max_covered, ifCovered(selected))
                return

            # Explore further columns for the combination
            for i in range(start, n - numSelect + len(selected) + 1):
                selected.append(i)  # Select the current column
                backtrack(i + 1, selected)  # Recurse to select more columns
                selected.pop()  # Backtrack by removing the last column

        # Initialize variables
        self.max_covered = 0
        m, n = len(matrix), len(matrix[0])

        # Precompute the number of `1`s in each row
        row_value = [sum(matrix[row]) for row in range(m)]

        # Start backtracking to select columns
        backtrack(0, [])
        return self.max_covered


class Solution:
    def maximumRows(self, matrix: List[List[int]], numSelect: int) -> int:
        # Backtracking function to generate all combinations of numSelect columns
        def backtrack(start: int, selected: List[int]) -> None:
            # Base case: if we have selected numSelect columns
            if len(selected) == numSelect:
                combinations.append(selected[:])  # Append a copy of the current combination
                return

            # Explore further columns
            for i in range(start, n - numSelect + len(selected) + 1):
                selected.append(i)  # Add the current column to the selection
                backtrack(i + 1, selected)  # Recurse to select the next column
                selected.pop()  # Backtrack by removing the last selected column

        max_covered = 0  # Variable to store the maximum number of covered rows
        combinations = []  # List to store all combinations of selected columns
        m, n = len(matrix), len(matrix[0])  # Dimensions of the matrix

        # Precompute row bitmasks to represent the columns required to cover each row
        row_masks = []
        for row in range(m):
            row_mask = 0
            for col in range(n):
                if matrix[row][col] == 1:
                    row_mask |= (1 << col)  # Set the bit for this column
            row_masks.append(row_mask)

        # Generate all possible combinations of numSelect columns
        backtrack(0, [])

        # Evaluate each combination to determine the number of covered rows
        for combination in combinations:
            # Create a bitmask for the selected combination of columns
            selected_mask = 0
            for col in combination:
                selected_mask |= (1 << col)  # Set the bit for each selected column

            # Count the number of rows fully covered by the selected columns
            covered = 0
            for row_mask in row_masks:
                if row_mask & selected_mask == row_mask:  # Check if all 1s in row_mask are covered
                    covered += 1

            # Update the maximum covered rows
            max_covered = max(max_covered, covered)

        return max_covered

```


### 131. Palindrome Partitioning

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        # Helper function to check if a substring is a palindrome
        def isPalindrome(string: str) -> bool:
            return string == string[::-1]

        # Backtracking function to find all palindrome partitions
        def backtrack(start: int, path: List[str]) -> None:
            # If we reach the end of the string, add the current partition to the result
            if start >= len(s):
                result.append(path[:])  # Append a copy of the current path
                return

            # Explore all substrings starting from 'start'
            for i in range(start, len(s)):
                substring = s[start:i+1]  # Current substring
                # Skip if it's not a palindrome
                if not isPalindrome(substring):
                    continue

                # Include the current substring and recurse
                path.append(substring)
                backtrack(i + 1, path)
                path.pop()  # Backtrack by removing the last added substring

        result = []  # To store all valid partitions
        backtrack(0, [])
        return result

```


### 2698. Find the Punishment Number of an Integer

```python
class Solution:
    def punishmentNumber(self, n: int) -> int:
        # Helper function to check if a number can be formed by splitting its square
        def backtrack(num: int, string: str, start: int, curSum: int) -> bool:
            # Base case: If we've processed the entire string, check if the sum matches the number
            if start >= len(string):
                return curSum == num

            # Try all possible splits of the string starting from 'start'
            for i in range(start, len(string)):
                # Extract the current substring and add it to the current sum
                value = curSum + int(string[start:i + 1])

                # If the sum exceeds the number, stop further exploration (pruning)
                if value > num:
                    break

                # Recurse to check the rest of the string with the updated sum
                if backtrack(num, string, i + 1, value):
                    return True

            # If no valid split is found, return False
            return False

        total = 0  # Initialize the total punishment number sum

        # Iterate through numbers from 1 to n
        for i in range(1, n + 1):
            square_str = str(i**2)  # Compute the square of the current number as a string

            # Check if the square can be split to sum up to the number
            if backtrack(i, square_str, 0, 0):
                total += i ** 2  # Add the square to the total if the condition is met

        return total  # Return the total punishment number

```

### 1593. Split a String Into the Max Number of Unique Substrings

```python
class Solution:
    def maxUniqueSplit(self, s: str) -> int:
        # Helper function to perform backtracking
        def backtrack(start: int, substrings: set) -> None:
            # Prune if the remaining characters + current substrings cannot exceed current maximum
            if len(s) - start + len(substrings) <= self.maximum:
                return

            # Base case: If we've processed the entire string, update the maximum
            if start == len(s):
                self.maximum = max(self.maximum, len(substrings))
                return

            # Iterate through all possible substrings starting from `start`
            for i in range(start, len(s)):
                substring = s[start:i+1]  # Extract the substring from `start` to `i`

                # Only proceed if the substring is not already used
                if substring not in substrings:
                    substrings.add(substring)  # Add the substring to the set
                    backtrack(i + 1, substrings)  # Recur with the next index
                    substrings.remove(substring)  # Backtrack and remove the substring

        self.maximum = 0  # Initialize the maximum number of unique substrings
        backtrack(0, set())  # Start backtracking from the beginning of the string
        return self.maximum  # Return the maximum unique split count

```

### Maximum Length of a Concatenated String with Unique Characters

```python
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        def isValid(string: str, chars: set) -> bool:
            for char in string:
                if char in chars:  # If a character is already in the set, it's invalid
                    return False
            return True  # All characters in `string` are unique with respect to `chars`

        def backtrack(start: int, chars: set) -> None:
            if start >= len(arr):  # Base case: all strings processed
                self.maximum = max(self.maximum, len(chars))  # Update maximum length
                return

            # Update the maximum length at the current state
            self.maximum = max(self.maximum, len(chars))

            # Try to include each string from `start` onward
            for i in range(start, len(arr)):
                # Skip this string if it cannot be added without duplicates
                if not isValid(arr[i], chars):
                    continue

                # Add characters of arr[i] to the current set
                for char in arr[i]:
                    chars.add(char)

                # Recursively backtrack with the next index
                backtrack(i + 1, chars)

                # Backtrack: Remove characters of arr[i] to try other combinations
                for char in arr[i]:
                    chars.remove(char)

        # Initialize the maximum length to 0
        self.maximum = 0

        # Pre-process `arr` to remove strings with duplicate internal characters
        arr = [string for string in arr if len(string) == len(set(string))]

        # Start backtracking with an empty set and the first index
        backtrack(0, set())

        # Return the maximum length of unique concatenation found
        return self.maximum

```

### 1849. Splitting a String Into Descending Consecutive Values

```python
class Solution:
    def splitString(self, s: str) -> bool:
        def backtrack(start: int, path: int, prev: int) -> bool:
            # Base case: If we have processed the entire string
            if start >= len(s):
                # Return True only if we have split into at least two parts
                return path > 1
            
            # Try splitting the string at every position from 'start'
            for i in range(start, len(s)):
                # Convert the current substring to a number
                num = int(s[start:i+1])

                # If prev is set, check if the difference is exactly 1
                if prev != -1 and prev - num != 1:
                    continue  # Skip if the condition is not met

                # Recursively check the remaining string
                if backtrack(i + 1, path + 1, num):
                    return True

            # Return False if no valid split is found
            return False

        # Start backtracking with initial values
        return backtrack(0, 0, -1)

```

### 2212. Maximum Points in an Archery Competition

```python
from typing import List

class Solution:
    def maximumBobPoints(self, numArrows: int, aliceArrows: List[int]) -> List[int]:
        def backtrack(index: int, arrows: int, curSum: int, path: List[int]) -> None:
            # Base case: If no more arrows or we've processed all sections
            if index == 12:
                # Update the maximum score and record Bob's arrow distribution
                if curSum > self.maximum:
                    self.maximum = curSum
                    self.bobArrows = path[:]
                    self.bobArrows[0] += arrows  # Assign remaining arrows to any section (default to 0)
                return

            # Option 1: Bob skips this section
            backtrack(index + 1, arrows, curSum, path)

            # Option 2: Bob wins this section
            if arrows > aliceArrows[index]:
                # Allocate enough arrows to win this section
                path[index] = aliceArrows[index] + 1
                backtrack(index + 1, arrows - path[index], curSum + index, path)
                path[index] = 0  # Backtrack to restore state

        # Initialize tracking variables
        self.maximum = 0
        self.bobArrows = [0] * 12

        # Start the backtracking process
        backtrack(0, numArrows, 0, [0] * 12)
        
        return self.bobArrows

```