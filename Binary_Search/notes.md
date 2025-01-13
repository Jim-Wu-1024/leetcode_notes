## Binary Search

1. Requirements for Binary Search
    - The array or list must be sorted.
    - The search operation divides the range in half repeatedly.

### 704. Binary Search

```python
from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # Special case: If the list has only one element, handle it separately
        if len(nums) == 1:
            return 0 if nums[0] == target else -1

        # Initialize pointers for the left and right bounds of the search range
        left, right = 0, len(nums) - 1

        # Perform the binary search
        while left <= right:
            # Calculate the middle index
            mid = (left + right) // 2

            # Check if the middle element is the target
            if nums[mid] == target:
                return mid

            # If the target is greater than the middle element,
            # narrow the search range to the right half
            if nums[mid] < target:
                left = mid + 1
            else:
                # Otherwise, narrow the search range to the left half
                right = mid - 1

        # If the loop ends, the target is not in the list
        return -1
```

### 35. Search Insert Position

Why return `left` not `right`: 

Returning `left` because at the end of the binary search, it represents the smallest index where the `target` can be inserted to maintain the sorted order of the array. In contrast, `right` does not reliably point to the correct insertion position, as it is always less than `left` when the loop ends.



```python
from typing import List

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        # Initialize pointers for the binary search
        left, right = 0, len(nums) - 1

        # Perform binary search to find the target or its insertion point
        while left <= right:
            mid = (left + right) // 2  # Calculate the middle index

            if nums[mid] == target:
                return mid  # Target found, return its index

            if nums[mid] < target:
                # Target is greater, search in the right half
                left = mid + 1
            else:
                # Target is smaller, search in the left half
                right = mid - 1

        # If target is not found, 'left' points to the correct insertion index
        return left

```

### 34. Find First and Last Position of Element in Sorted Array

1. **Binary Search**:
   - Use binary search to efficiently locate the `target` in \(O(log n)\) time.
   - Perform two separate binary searches:
     - One for the **first occurrence** of the `target`.
     - One for the **last occurrence** of the `target`.

2. **Logic for Binary Search**:
    - If `nums[mid] == target`:
       - Update `index` to `mid`.
       - If `first` is `True`, search left (`right = mid - 1`) for the first occurrence.
       - If `first` is `False`, search right (`left = mid + 1`) for the last occurrence.


```python
from typing import List

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # Helper function to perform binary search
        def binarySearch(nums: List[int], target: int, first: bool=False) -> int:
            left, right = 0, len(nums) - 1
            index = -1  # Default value if target is not found
            
            # Perform binary search
            while left <= right:
                mid = (left + right) // 2

                if nums[mid] < target:
                    # Target is larger, move to the right half
                    left = mid + 1
                elif nums[mid] > target:
                    # Target is smaller, move to the left half
                    right = mid - 1
                else:
                    # Target found, update index
                    index = mid
                    if first:
                        # Continue searching the left half for the first occurrence
                        right = mid - 1
                    else:
                        # Continue searching the right half for the last occurrence
                        left = mid + 1
                
            return index

        # Edge case: empty array
        if not nums:
            return [-1, -1]

        # Find the first and last occurrences of the target
        first = binarySearch(nums, target, True)
        last = binarySearch(nums, target, False)

        return [first, last]

```

### 69. Sqrt(x)

The square root of any number *x* is always less than or equal to *x//2*.

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        # Edge case: if x is 0 or 1, the square root is x itself
        if x < 2:
            return x

        # Initialize the binary search range
        # We only need to search up to x // 2 because the square root of x is always <= x // 2 for x >= 2
        left, right = 0, x // 2

        # Perform binary search to find the integer square root
        while left <= right:
            mid = (left + right) // 2  # Calculate the middle of the current range

            # Check if mid squared is exactly x
            if mid ** 2 == x:
                return mid  # Exact square root found

            # If mid squared is less than x, move to the right half
            if mid ** 2 < x:
                left = mid + 1
            else:
                # If mid squared is greater than x, move to the left half
                right = mid - 1

        # When the loop ends, 'right' is the largest integer such that right^2 <= x
        return right

```

### 367. Valid Perfect Square

```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        # Edge case: handle numbers less than 2
        if num < 2:
            return True  # 0 and 1 are perfect squares

        # Initialize binary search range
        left, right = 0, num // 2

        # Perform binary search
        while left <= right:
            mid = left + (right - left) // 2  # Avoid overflow
            mid_square = mid ** 2  # Compute square of mid

            if mid_square == num:
                return True  # Found a perfect square
            elif mid_square < num:
                left = mid + 1  # Search in the right half
            else:
                right = mid - 1  # Search in the left half

        # If no perfect square found, return False
        return False

```

### Minimum Time to Complete Trips

```python
from typing import List

from typing import List

class Solution:
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        def calculateTotalTrips(avail_time: int, time: List[int]) -> int:
            # For each bus, calculate how many trips it can complete in avail_time
            # and sum them up.
            return sum(avail_time // t for t in time)

        # Initialize the binary search range:
        # The minimum time required is 0, and the maximum is when the fastest bus
        # makes all totalTrips.
        left, right = 0, min(time) * totalTrips

        # Perform binary search to find the minimum time required
        while left <= right:
            mid = (left + right) // 2  # Calculate the midpoint of the current range

            # Calculate the total trips that can be completed in 'mid' time
            total = calculateTotalTrips(mid, time)

            if total < totalTrips:
                # If not enough trips are possible, increase the time
                left = mid + 1
            else:
                # If enough or more trips are possible, try to reduce the time
                right = mid - 1

        # After the loop, 'left' will be the smallest time at which all totalTrips
        # can be completed
        return left

```

### Range Frequency Queries

```python
from typing import List

class RangeFreqQuery:
    def __init__(self, arr: List[int]):
        self.freq = {}  # Dictionary to store the indices of each value in the array
        for i, num in enumerate(arr):
            if num not in self.freq:
                self.freq[num] = []  # Initialize an empty list for a new value
            self.freq[num].append(i)  # Append the index of the current value to its list

    def query(self, left: int, right: int, value: int) -> int:
        def binarySearch(indices: List[int], target: int, find_left: bool) -> int:
            l, r = 0, len(indices) - 1
            index = -1  # Default value if no valid index is found
            
            while l <= r:
                mid = (l + r) // 2  # Calculate the middle index

                if indices[mid] == target:
                    return mid  # Return immediately if the exact match is found

                if indices[mid] < target:
                    if not find_left:
                        index = mid  # Update index for rightmost search
                    l = mid + 1  # Narrow search to the right half
                else:
                    if find_left:
                        index = mid  # Update index for leftmost search
                    r = mid - 1  # Narrow search to the left half
            
            return index

        # If the value is not in the array, its frequency is 0
        if value not in self.freq:
            return 0

        # Get the list of indices where the value appears
        indices = self.freq[value]

        # Find the first index >= left and the last index <= right
        start = binarySearch(indices, left, True)
        end = binarySearch(indices, right, False)

        # Check if the indices are valid and within range
        if start == -1 or end == -1 or start > end:
            return 0

        # Return the count of occurrences in the range
        return end - start + 1
        
```

### House Robber IV

```python
from typing import List

class Solution:
    def minCapability(self, nums: List[int], k: int) -> int:
        def atLeastK(nums: List[int], capacity: int, k: int) -> bool:
            # Dynamic programming array: dp[i][0] means the max count without choosing nums[i]
            # dp[i][1] means the max count with choosing nums[i]
            dp = [[0] * 2 for _ in range(len(nums))]

            # Initialize the first element of dp
            dp[0][0] = 0  # Not choosing the first element
            dp[0][1] = 1 if nums[0] <= capacity else 0  # Choosing the first element if valid

            # Fill the dp array
            for i in range(1, len(nums)):
                # If not choosing nums[i], take the max count from previous state
                dp[i][0] = max(dp[i-1][0], dp[i-1][1])
                # If choosing nums[i], add 1 to the count of the previous "not chosen" state
                if nums[i] <= capacity:
                    dp[i][1] = dp[i-1][0] + 1

            # Return True if we can pick at least 'k' elements
            return max(dp[-1][0], dp[-1][1]) >= k

        # Edge case: if there's only one element, return it as the minimum capability
        if len(nums) == 1:
            return nums[0]

        # Binary search over the range of possible capabilities
        left, right = min(nums), max(nums)

        while left <= right:
            mid = (left + right) // 2  # Middle point of the current search range

            # Check if we can achieve the desired condition with the current mid value
            if not atLeastK(nums, mid, k):
                left = mid + 1  # Increase the minimum capability
            else:
                right = mid - 1  # Decrease the maximum capability

        # Return the smallest capability that allows at least 'k' elements to be chosen
        return left

```

### Maximum Candies Allocated to K Children

```python
from typing import List

class Solution:
    def maximumCandies(self, candies: List[int], k: int) -> int:
        def canDistribute(pile_size: int) -> bool:
            """
            Check if we can create at least `k` piles of size `pile_size` 
            from the given candies.
            """
            total_piles = sum(candy // pile_size for candy in candies)
            return total_piles >= k

        # If there aren't enough candies to create `k` piles, return 0.
        if sum(candies) < k:
            return 0

        # Binary search to find the maximum pile size.
        left, right = 1, max(candies)
        while left <= right:
            mid = (left + right) // 2
            if canDistribute(mid):
                left = mid + 1  # Try for a larger pile size
            else:
                right = mid - 1  # Reduce the pile size

        return right

```

### Maximize Score of Numbers in Ranges

```python
from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        def canAchieve(start: List[int], diff: int) -> bool:
            # Start by picking the smallest value from the first interval
            element = start[0]

            # Iterate through the remaining intervals
            for i in range(1, n):
                # Select the smallest valid number from the current interval
                element = max(element + diff, start[i])
                
                # If the chosen number exceeds the current interval's end, the target diff is not achievable
                if element > start[i] + d:
                    return False

            # If all intervals are satisfied, the target diff is achievable
            return True

        # Number of intervals
        n = len(start)

        # Sort intervals to process them in increasing order of starting points
        start.sort()

        # Binary search to find the maximum achievable minimum difference
        left, right = 0, (start[-1] + d) - start[0]
        while left <= right:
            mid = (left + right) // 2  # Middle value to test as the minimum difference

            if canAchieve(start, mid):
                # If the current diff is achievable, try for a larger diff
                left = mid + 1
            else:
                # If not achievable, reduce the target diff
                right = mid - 1

        # The largest achievable diff is stored in `right`
        return right

```

### 1818. Minimum Absolute Sum Difference

```python
from typing import List

class Solution:
    def minAbsoluteSumDiff(self, nums1: List[int], nums2: List[int]) -> int:
        def search_closest(nums: List[int], target: int) -> int:
            # Binary search to find the closest value to target
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] == target:
                    return 0  # Exact match, no difference
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1

            # Determine the closest value between nums[left] and nums[left - 1]
            if left == 0:
                return abs(nums[left] - target)
            if left >= len(nums):
                return abs(nums[left - 1] - target)

            return min(abs(nums[left] - target), abs(nums[left - 1] - target))

        MOD = 10**9 + 7

        # Step 1: Sort nums1 for binary search
        search = sorted(nums1)

        total_diff = 0
        max_gain = 0

        # Step 2: Calculate the total difference and potential maximum gain
        for i in range(len(nums1)):
            diff = abs(nums1[i] - nums2[i])
            total_diff += diff

            # Find the potential smallest difference if nums1[i] is replaced
            closest_diff = search_closest(search, nums2[i])
            gain = diff - closest_diff

            # Track the maximum gain
            max_gain = max(max_gain, gain)

        # Step 3: Return the minimized total difference
        return (total_diff - max_gain) % MOD

```