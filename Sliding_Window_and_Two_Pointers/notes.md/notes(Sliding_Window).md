## Sliding Window and Two Pointers

### 1.  Sliding Window

**When to Use:**
- Problems with contiguous elements (subarrays, substrings).
- Optimization problems (e.g., smallest/largest subarray, sum, etc.).
- Dynamic constraints (e.g., unique elements, target sum).

**How It Works:**
1. **Two Pointers**: Use `left` and `right` to represent the window boundaries.
2. **Expand and Shrink**:
   - Expand by moving `right`.
   - Shrink by moving `left` when constraints are met.
3. **Condition Check**: Track whether the window satisfies the problem's requirements.
4. **Result Update**: Update the best result (e.g., min length, max sum) when applicable.


### 209. Minimum Size Subarray Sum

```python
from typing import List

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        # Initialize the left pointer for the sliding window
        left = 0

        # Initialize total to keep track of the sum of the current subarray
        # Initialize length to infinity to find the minimum subarray length
        total, length = 0, float('inf')

        # Iterate through the array using the right pointer
        for right in range(len(nums)):
            # Add the current element to the total
            total += nums[right]
            
            # Check if the current total meets or exceeds the target
            while total >= target:
                # Update the minimum length of the subarray
                length = min(length, right - left + 1)
                # Shrink the window from the left
                total -= nums[left]
                left += 1

        # If no valid subarray was found, return 0; otherwise, return the minimum length
        return length if length != float('inf') else 0

```


### Fruits into Baskets

```python
from typing import List

class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        left = 0
        types = {}
        maximum = 0

        for right in range(len(fruits)):
            # Add the current fruit to the window
            types[fruits[right]] = types.get(fruits[right], 0) + 1

            # Shrink the window if there are more than 2 types of fruits
            while len(types) > 2:
                types[fruits[left]] -= 1
                if types[fruits[left]] == 0:
                    types.pop([fruits[left]])
                left += 1

            # Update the maximum fruits collected in a valid window
            maximum = max(maximum, right - left + 1)

        return maximum    

```

### Minimum Window Substring


```python
from collections import Counter

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(t) > len(s):
            return ''

        # Count the frequency of each character in t
        needed = Counter(t)
        required = len(needed)  # Number of unique characters in t that must be matched

        # Sliding window pointers and tracking variables
        left = 0
        formed = 0  # Number of unique characters in the current window matching the count in t
        cur = {}  # Current window character frequencies
        min_len = float('inf') 
        min_start = 0 

        # Expand the window by moving the right pointer
        for right in range(len(s)):
            char = s[right]
            if char in needed:
                # Add the current character to the current window's frequency count
                cur[char] = cur.get(char, 0) + 1

                # If the character's frequency matches the required frequency, update `formed`
                if cur[char] == needed[char]:
                    formed += 1

            # Contract the window from the left if all characters are matched
            while formed == required:
                # Update the minimum window size and its starting index
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                    min_start = left

                # Remove the leftmost character from the window
                char_left = s[left]
                if char_left in needed:
                    cur[char_left] -= 1
                    # If the character's frequency in the window falls below the required frequency
                    if cur[char_left] < needed[char_left]:
                        formed -= 1

                # Move the left pointer to shrink the window
                left += 1

        # Return the smallest window or an empty string if no valid window is found
        return s[min_start:min_start + min_len] if min_len != float('inf') else ''


```

### 1358. Number of Substrings Containing All Three Characters

- Use two pointers (`left` and `right`) to represent the current window.
- Use a dictionary (`freq`) to store the frequency of each character in the current window.
- For every valid window, count all substrings starting from the current `left` and ending at or after `right`.
   - Add `(n - right)` to the count, where `n` is the length of the string.

```python
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)

        left = 0
        count = 0
        freq = {}

        for right in range(n):
            char = s[right]
            freq[char] = freq.get(char, 0) + 1

            # Shrink the window when all three characters are present
            while len(freq) == 3:
                # Count all valid substrings starting at `left`
                count += (n - right)

                # Remove the leftmost character from the window
                char_left = s[left]
                freq[char_left] -= 1
                if freq[char_left] == 0:
                    del freq[char_left]
                left += 1

        return count

```

### 1343. Number of Subarrays of Size k and Average Greater than or Equal to Threshold

```python
from typing import List

class Solution:
    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        left = 0
        total = 0
        count = 0

        for right in range(len(arr)):
            total += arr[right]  # Expand the window by adding the right element

            # When the window size reaches `k`
            if right - left + 1 == k:
                if total / k >= threshold:  # Check if the average meets the threshold
                    count += 1

                total -= arr[left]  # Shrink the window by removing the left element
                left += 1

        return count

```


### 1248. Count Number of Nice Subarrays

- Expand the window by moving `right`.
- Shrink the window by moving `left` to ensure at most `k` odd numbers are in the window.

 - When the window contains exactly `k` odd numbers:
     - Count the number of valid subarrays ending at the current `right`.
     - Use a variable `even_prefix` to count even numbers to the left of the window, which extend the valid subarrays.

```python
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        left = 0

        odd_count = 0
        result = 0
        even_prefix = 0  # Counts the number of even numbers before the current window
        for right in range(len(nums)):
            # Increment odd count if the current number is odd
            if nums[right] % 2 != 0:
                odd_count += 1
                even_prefix = 0  # Reset even_prefix when a new odd is added

            # Shrink the window until there are exactly `k` odd numbers
            while odd_count > k:
                if nums[left] % 2 != 0:
                    odd_count -= 1
                left += 1

            # If the current window contains exactly `k` odd numbers
            if odd_count == k:
                # Count even numbers to the left of the window
                while nums[left] % 2 == 0:
                    even_prefix += 1
                    left += 1
                # Add valid subarrays to the result
                result += even_prefix + 1

        return result

```

### 2461. Maximum Sum of Distinct Subarray With Length K

```python
from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        left = 0
        total = 0
        unique = dict()
        maximum = 0

        for right in range(len(nums)):
            # Add current element to the total and count in the dictionary
            unique[nums[right]] = unique.get(nums[right], 0) + 1
            total += nums[right]

            # If the window size reaches `k`
            if right - left + 1 == k:
                # Check if all elements in the window are unique
                if len(unique) == k:
                    maximum = max(maximum, total)

                # Slide the window by removing the left element
                total -= nums[left]
                unique[nums[left]] -= 1
                if unique[nums[left]] == 0:
                    del unique[nums[left]]

                left += 1

        return maximum


class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        # Initialize the left pointer, current sum, and maximum sum
        left = 0
        total = 0
        maximum = 0
        # Set to track unique elements in the current window
        unique = set()

        # Iterate over the array using the right pointer
        for right in range(len(nums)):
            # If the current element is already in the set, shrink the window
            while nums[right] in unique:
                # Remove the leftmost element from the set and update the total sum
                unique.remove(nums[left])
                total -= nums[left]
                # Move the left pointer to the right
                left += 1

            # Add the current element to the set and update the total sum
            unique.add(nums[right])
            total += nums[right]

            # Check if the window size matches `k`
            if right - left + 1 == k:
                # Update the maximum sum if all elements in the window are unique
                maximum = max(maximum, total)
                # Shrink the window from the left to maintain size `k`
                unique.remove(nums[left])
                total -= nums[left]
                left += 1

        # Return the maximum sum found
        return maximum

```

### 2024. Maximize the Confusion of an Exam

```python
class Solution:
    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        left = 0
        ans_freq = {'T': 0, 'F': 0}
        maximum = 0

        for right in range(len(answerKey)):
            # Increment the count for the current character
            ans_freq[answerKey[right]] += 1

            # Ensure the smaller frequency of 'T' or 'F' does not exceed k
            while min(ans_freq.values()) > k:
                ans_freq[answerKey[left]] -= 1
                left += 1

            # Update the maximum length of the valid window
            maximum = max(maximum, right - left + 1)
        
        return maximum

```
