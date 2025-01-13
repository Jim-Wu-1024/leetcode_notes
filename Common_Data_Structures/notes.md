### 2874. Maximum Value of an  Ordered Triplet II

- The `prefix_max` array tracks the **maximum value to the left of each index j**, avoiding the need to repeatedly iterate through the array to find left-side maximums.
- Similarly, the `suffix_max` array tracks the **maximum value to the right of each index j**.

```python
from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        maximum = 0  # Initialize the maximum value to 0

        # Step 1: Compute prefix maximums
        # prefix_max[j] will store the largest value to the left of index j
        prefix_max = [float('-inf')] * n
        for j in range(1, n):
            prefix_max[j] = max(prefix_max[j-1], nums[j-1])

        # Step 2: Compute suffix maximums
        # suffix_max[j] will store the largest value to the right of index j
        suffix_max = [float('-inf')] * n
        for j in range(n-2, -1, -1):
            suffix_max[j] = max(suffix_max[j+1], nums[j+1])

        # Step 3: Calculate the maximum triplet value
        # For each middle element nums[j], check the triplet condition
        for j in range(1, n-1):
            # Only compute if prefix_max[j] > nums[j] and nums[j] > suffix_max[j]
            maximum = max(maximum, (prefix_max[j] - nums[j]) * suffix_max[j])

        return maximum

```

### 1010. Pairs of Songs With Total Durations Divisible by 60

```python
from typing import List

class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        # Step 1: Count frequencies of remainders modulo 60
        modulo = [0] * 60
        for t in time:
            modulo[t % 60] += 1

        # Step 2: Calculate pairs
        pair = 0

        # Pairs where remainder is 0
        pair += (modulo[0] * (modulo[0] - 1)) // 2

        # Pairs where remainder is 30 (special case)
        pair += (modulo[30] * (modulo[30] - 1)) // 2

        # Pairs for other remainders (1-29)
        for i in range(1, 30):
            pair += modulo[i] * modulo[60 - i]

        return pair

```

### Unique Length-3 Palindromic Subsequences

```python
class Solution:
    def countPalindromicSubsequence(self, s: str) -> int:
        # Step 1: Track the first and last occurrence of each character
        first_occurrence = {}
        last_occurrence = {}
        for i, char in enumerate(s):
            if char not in first_occurrence:
                first_occurrence[char] = i
            last_occurrence[char] = i

        # Step 2: Count unique palindromic subsequences
        unique_count = 0
        for char, left in first_occurrence.items():
            right = last_occurrence[char]
            if right - left > 1:  # Ensure at least one character between the two
                # Count unique characters in the range (left+1, right-1)
                middle_chars = set(s[left + 1:right])
                unique_count += len(middle_chars)

        return unique_count

```

### 1014. Best Sightseeing Pair

```python
class Solution:
    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        maximum = 0  # To track the maximum score
        i_max = values[0]  # Maximum value of values[i] + i so far

        for j in range(1, len(values)):
            # Calculate the score for the current pair (i, j)
            maximum = max(maximum, i_max + values[j] - j)

            # Update i_max to include the current value[j] + j
            i_max = max(i_max, values[j] + j)

        return maximum

```

### 2559. Count Vowel Strings in Ranges

```python
from typing import List

class Solution:
    def vowelStrings(self, words: List[str], queries: List[List[int]]) -> List[int]:
        # Define vowels as a set for quick lookup
        vowel = {'a', 'e', 'i', 'o', 'u'}

        # Helper function to check if a word is valid
        def isValid(word: str) -> bool:
            return word[0] in vowel and word[-1] in vowel

        # Step 1: Precompute prefix sums
        n = len(words)
        prefix = [0] * (n + 1)
        for i, word in enumerate(words):
            prefix[i + 1] = prefix[i] + (1 if isValid(word) else 0)

        # Step 2: Process queries
        result = []
        for l, r in queries:
            result.append(prefix[r + 1] - prefix[l])

        return result

```

### 3152. Special Array II

```python
from typing import List

class Solution:
    def isArraySpecial(self, nums: List[int], queries: List[List[int]]) -> List[bool]:
        # Step 1: Initialize a prefix array where each index will track
        # the cumulative count of valid parity transitions.
        prefix = [1] * len(nums)  # Start with all indices initialized to 1

        # Step 2: Build the prefix array based on parity transitions
        for i in range(1, len(nums)):
            prefix[i] = prefix[i - 1] + 1 if nums[i-1] % 2 != nums[i] % 2 else prefix[i]


        # Step 3: Process each query
        result = []
        for l, r in queries:
            # If the query is a single-element subarray, it is always "special"
            if l == r:
                result.append(True)
                continue

            # Use the prefix array to calculate the length of valid transitions
            # in the range [l, r]. If the length is at least the size of the
            # subarray, the subarray is "special".
            length = prefix[r]
            result.append(length >= (r - l + 1))
        
        # Step 4: Return the results for all queries
        return result

```

### 2438. Range Product Queries of Powers

```python
from typing import List

class Solution:
    def productQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        MOD = 10**9 + 7

        powers = []
        power = 0
        # Step 1: Extract powers of 2 from binary representation of n
        while n > 0:
            if n & 1 != 0:
                powers.append(power)
            
            power += 1
            n >>= 1

        prefix_sum = [0] * len(powers)
        prefix_sum[0] = powers[0]
        # Step 2: Compute prefix sum of exponents
        for i in range(1, len(powers)):
            prefix_sum[i] = prefix_sum[i-1] + powers[i]

        result = []
        for l, r in queries:
            # Compute the total exponent for the range [l, r]
            result.append((2 ** (prefix_sum[r] - (prefix_sum[l-1] if l-1 >= 0 else 0)) % MOD))

        return result

```

### 1749. Maximum Absolute Sum of Any Subarray

**Kadane's Algorithm:** 
- Iterative Calculation: Traverse the array while maintaining a running sum of the current subarray.

- Dynamic Choice:
   - At each step, decide whether to:
      - Continue the current subarray (current_sum + num), or
      - Start a new subarray from the current element (num alone).

```python
from typing import List

class Solution:
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        # Initialize variables
        current_max, global_max = 0, float('-inf')
        current_min, global_min = 0, float('inf')
        
        # Iterate through the array
        for num in nums:
            current_max = max(num, current_max + num)
            current_min = min(num, current_min + num)
            
            global_max = max(global_max, current_max)
            global_min = min(global_min, current_min)
        
        # Return the maximum absolute value between global max and min
        return max(abs(global_max), abs(global_min))

```

### 930. Binary Subarrays With Sum / 560. Subarray Sum Equals K

**Key Ideas:** 

$$ sum(i,j)=sum(0,j)−sum(0,i) => cur\_sum − k = prefix sum of an earlier subarray $$ 

```python
from typing import List

class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        # Initialize prefix sum count map and variables
        prefix_count = {0: 1}  # To handle subarrays starting from index 0
        current_sum = 0
        count = 0
        
        # Traverse the array
        for num in nums:
            current_sum += num  # Update running sum
            
            # Check if there's a prefix sum that matches (current_sum - goal)
            if current_sum - goal in prefix_count:
                count += prefix_count[current_sum - goal]
            
            # Update prefix_count with current_sum
            prefix_count[current_sum] = prefix_count.get(current_sum, 0) + 1
        
        return count


class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # Initialize hash map to store prefix sum frequencies
        prefix_sum_count = {0: 1}  # To account for subarrays starting at index 0
        current_sum = 0
        count = 0
        
        # Iterate through the array
        for num in nums:
            current_sum += num  # Update running sum
            
            # Check if (current_sum - k) exists in prefix_sum_count
            if current_sum - k in prefix_sum_count:
                count += prefix_sum_count[current_sum - k]
            
            # Update prefix_sum_count with the current_sum
            prefix_sum_count[current_sum] = prefix_sum_count.get(current_sum, 0) + 1
        
        return count


```

### 1524. Number of Sub-arrays With Odd Sum

A subarray has an odd sum if the cumulative prefix sum (from the start to the current index) minus a previous prefix sum is odd.

This means we need to track the parity (odd or even) of the prefix sums.

```python
class Solution:
    def numOfSubarrays(self, arr: List[int]) -> int:
        MOD = 10**9 + 7
        # Track counts of even and odd prefix sums
        prefix_parity = {0: 1, 1: 0}
        
        cur_sum = 0
        count = 0
        
        for num in arr:
            cur_sum += num  # Update prefix sum
            
            if cur_sum % 2 == 1:  # Current prefix sum is odd
                count += prefix_parity[0]  # Add count of even prefix sums
            else:  # Current prefix sum is even
                count += prefix_parity[1]  # Add count of odd prefix sums
            
            # Increment the count of the current parity in the map
            prefix_parity[cur_sum % 2] += 1
            
            # Apply modulo to count
            count %= MOD
        
        return count

```

## Stack (First In Last Out) / Queue (First In First Out)

### 1441. Build an Array With Stack Operations

```python
from typing import List

class Solution:
    def buildArray(self, target: List[int], n: int) -> List[str]:
        # Initialize the list to store operations
        operations = []
        # Pointer to track the current index in the target array
        index = 0

        # Iterate through numbers from 1 to n
        for i in range(1, n + 1):
            # If all elements of target are processed, stop the loop
            if index >= len(target):
                break

            # If the current number matches the current element in target
            if i == target[index]:
                operations.append('Push')  # Push the number to the stack
                index += 1  # Move to the next element in target
            else:
                # If the current number is not in target, simulate a "Push" followed by a "Pop"
                operations.append('Push')  # Push the number to the stack
                operations.append('Pop')  # Immediately pop it off

        return operations

```

### 946. Validate Stack Sequences

- After pushing an element from pushed, check if it matches the current popped element. If yes, pop it.

```python
from typing import List

class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        index = 0  # Pointer for the popped sequence

        for value in pushed:
            stack.append(value)  # Simulate a push operation
            
            # While the stack is not empty and the top matches the current popped element
            while stack and stack[-1] == popped[index]:
                stack.pop()  # Simulate a pop operation
                index += 1  # Move to the next element in popped

        # If all elements in popped are matched, the stack should be empty
        return not stack

```

### 1472. Design Browser History

```python
class BrowserHistory:

    def __init__(self, homepage: str):
        self.history = [homepage]  # Store browsing history
        self.index = 0  # Current position in history

    def visit(self, url: str) -> None:
        # Discard forward history and append the new URL
        self.history = self.history[:self.index + 1]
        self.history.append(url)
        self.index += 1

    def back(self, steps: int) -> str:
        # Move back up to the beginning of the history
        self.index = max(0, self.index - steps)
        return self.history[self.index]

    def forward(self, steps: int) -> str:
        # Move forward up to the most recent URL
        self.index = min(len(self.history) - 1, self.index + steps)
        return self.history[self.index]

```

### 232. Implement Queue using Stacks

- **Two Stacks**:
  - `pushed`: Stores elements for enqueue operations.
  - `popped`: Handles dequeue and peek operations.

- **Transfer Rule**:
  - Transfer all elements from `pushed` to `popped` **only when `popped` is empty**.

- **Amortized \(O(1)\)**:
  - Each element is transferred at most once between stacks, ensuring \(O(1)\) average time per operation.

- **Operations**:
  - **`push(x)`**: Append to `pushed` (\(O(1)\)).
  - **`pop()`**: Transfer if `popped` is empty, then pop from `popped`.
  - **`peek()`**: Transfer if `popped` is empty, then return the top of `popped`.
  - **`empty()`**: Return `True` if both stacks are empty.

- **Efficiency**:
  - \(O(1)\) amortized time for all operations.


```python
class MyQueue:

    def __init__(self):
        self.pushed = []  # Stack to handle enqueue operations
        self.popped = []  # Stack to handle dequeue operations

    def push(self, x: int) -> None:
        self.pushed.append(x)  # Push to the enqueue stack

    def pop(self) -> int:
        # Transfer elements to popped stack if empty
        if not self.popped:
            while self.pushed:
                self.popped.append(self.pushed.pop())
        return self.popped.pop()  # Pop from the dequeue stack

    def peek(self) -> int:
        # Transfer elements to popped stack if empty
        if not self.popped:
            while self.pushed:
                self.popped.append(self.pushed.pop())
        return self.popped[-1]  # Peek the top of the dequeue stack

    def empty(self) -> bool:
        # Queue is empty if both stacks are empty
        return not self.pushed and not self.popped

```

### 225. Implement Stack using Queues

```python
from collections import deque

class MyStack:

    def __init__(self):
        self.queue = deque()

    def push(self, x: int) -> None:
        self.queue.append(x)

    def pop(self) -> int:
        # Rotate n-1 elements to the back
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())
        return self.queue.popleft()  # Remove and return the last element

    def top(self) -> int:
        # Rotate n-1 elements to the back
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())
        top_element = self.queue[0]  # Retrieve the last element
        self.queue.append(self.queue.popleft())  # Restore the queue order
        return top_element

    def empty(self) -> bool:
        return not self.queue

```

### 347. Top K Frequent Elements

```python
from typing import List
from collections import Counter
import heapq

class Solution:
    """
    Heap-based Solution: O(nlogk)
    """
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # Create a min-heap to store the k most frequent elements
        heap = []
        # Count the frequency of each number in the input list
        freq = Counter(nums)

        # Iterate through the frequency dictionary
        for num, count in freq.items():
            # If the heap already has k elements
            if len(heap) >= k:
                # Compare the frequency of the current number with the smallest frequency in the heap
                if count > heap[0][0]:
                    # Remove the smallest frequency element from the heap
                    heapq.heappop(heap)
                    # Add the current element to the heap
                    heapq.heappush(heap, (count, num))
            else:
                # If the heap has fewer than k elements, directly add the current element
                heapq.heappush(heap, (count, num))

        # Extract the numbers from the heap and return them as the result
        return [num for count, num in heap]


class Solution:
    """
    Bucket-based Solution: O(n), If k is small relative to the input size, bucket sort can provide a more efficient solution.
    
    Use an array (bucket) where the index represents the frequency, and the values are lists of numbers with that frequency.
    """
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # Count frequencies
        freq = Counter(nums)
        # Create buckets where index represents frequency
        buckets = [[] for _ in range(len(nums) + 1)]
        for num, count in freq.items():
            buckets[count].append(num)
        
        # Gather top k frequent elements
        result = []
        for i in range(len(buckets) - 1, 0, -1):
            for num in buckets[i]:
                result.append(num)
                if len(result) == k:
                    return result

```

### 239. Sliding Window Maximum / 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

**239:**
  
  Deque for Maximum Tracking:

- Use a deque to store indices of elements in the current window.
- The deque maintains elements in decreasing order of value:
   - The front of the deque always holds the index of the maximum element in the window.
   - Smaller elements are removed from the deque because they can never be the maximum while larger elements exist.

**1438:**

Use two deques to track the maximum and minimum values in the current window:
  - `max_queue`: Stores indices of elements in decreasing order of their values.
  - `min_queue`: Stores indices of elements in increasing order of their values.

```python
from typing import List
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if k == 1:  # Special case: window size 1, all elements are maxima
            return nums

        left, right = 0, 0  # Initialize two pointers for the sliding window
        queue = deque()  # Deque to store (value, index) in descending order of value
        result = []  # List to store the maximum of each window

        while right < len(nums):
            # Remove elements from the back of the deque if they are smaller than the current element
            while queue and nums[right] > queue[-1][0]:
                queue.pop()
            
            # Add the current element and its index to the deque
            queue.append((nums[right], right))

            # Check if the current window has reached the desired size
            if right - left + 1 == k:
                # Append the maximum value (front of the deque) to the result
                result.append(queue[0][0])

                # Move the left pointer to shrink the window
                left += 1

                # Remove elements from the front of the deque if they are out of the current window
                if left > queue[0][1]:
                    queue.popleft()
            
            # Expand the window by moving the right pointer
            right += 1
        
        return result


from collections import deque
from typing import List

class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        left = 0
        max_len = 0
        max_queue, min_queue = deque(), deque()

        for right in range(len(nums)):
            # Maintain decreasing order in max_queue
            while max_queue and nums[right] > nums[max_queue[-1]]:
                max_queue.pop()
            max_queue.append(right)

            # Maintain increasing order in min_queue
            while min_queue and nums[right] < nums[min_queue[-1]]:
                min_queue.pop()
            min_queue.append(right)

            # Check if the current window is valid
            while nums[max_queue[0]] - nums[min_queue[0]] > limit:
                left += 1
                # Remove elements out of the window from the deques
                if max_queue[0] < left:
                    max_queue.popleft()
                if min_queue[0] < left:
                    min_queue.popleft()

            # Update the maximum length of the valid window
            max_len = max(max_len, right - left + 1)

        return max_len


```

### 1209. Remove All Adjacent Duplicates in String II

```python
class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        stack = []  # Stack to store (character, count)

        for char in s:
            if stack and stack[-1][0] == char:
                # Increment the count of the top element
                stack[-1] = (char, stack[-1][1] + 1)
                # Remove the element if the count reaches k
                if stack[-1][1] == k:
                    stack.pop()
            else:
                # Push a new character onto the stack with count 1
                stack.append((char, 1))

        # Reconstruct the result string directly
        return ''.join(char * count for char, count in stack)

```

### 950. Reveal Cards In Increasing Order

- **Simulate Reveal Order:**

   - Use a deque to represent indices of the result array.
   - Rotate indices in the deque to simulate revealing the top card and placing the next card at the bottom.

- **Place Cards:**

   - For each card in the sorted deck:
      1. Place the card at the index from the front of the deque.
      2. Move the next index in the deque to the back.

```python
from collections import deque
from typing import List

class Solution:
    def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
        # Step 1: Sort the deck in ascending order
        deck.sort()

        # Step 2: Create a queue to simulate the original indices
        queue = deque(range(len(deck)))

        # Step 3: Prepare the result array
        result = [0] * len(deck)

        # Step 4: Place cards in the simulated order
        for card in deck:
            # Place the smallest card at the index from the queue
            result[queue.popleft()] = card
            # Move the next index to the back of the queue (if any)
            if queue:
                queue.append(queue.popleft())

        return result

```

### 1670. Design Front Middle Back Queue

```python
from collections import deque

class FrontMiddleBackQueue:
    def __init__(self):
        self.front = deque()
        self.back = deque()

    def _balance(self):
        """Ensure that the front deque has at most one more element than the back deque."""
        while len(self.front) > len(self.back) + 1:
            self.back.appendleft(self.front.pop())
        while len(self.back) > len(self.front):
            self.front.append(self.back.popleft())

    def pushFront(self, val: int) -> None:
        self.front.appendleft(val)
        self._balance()

    def pushMiddle(self, val: int) -> None:
        if len(self.front) > len(self.back):
            self.back.appendleft(self.front.pop())
        self.front.append(val)

    def pushBack(self, val: int) -> None:
        self.back.append(val)
        self._balance()

    def popFront(self) -> int:
        if not self.front and not self.back:
            return -1
        value = self.front.popleft()
        self._balance()
        return value

    def popMiddle(self) -> int:
        if not self.front and not self.back:
            return -1
        value = self.front.pop()
        self._balance()
        return value

    def popBack(self) -> int:
        if not self.front and not self.back:
            return -1
        value = self.back.pop() if self.back else self.front.pop()
        self._balance()
        return value
        
```

## Heap

- A **heap** is a specialized binary tree-based data structure.
- It satisfies the heap property:
   - **Max-Heap**: Every parent node is greater than or equal to its child nodes.
   - **Min-Heap**: Every parent node is less than or equal to its child nodes.

### 1046. Last Stone Weight

```python
import heapq
from typing import List

class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        # Step 1: Create a max-heap by pushing negative weights
        heap = [-stone for stone in stones]
        heapq.heapify(heap)

        # Step 2: Smash stones until at most one remains
        while len(heap) > 1:
            # Pop the two heaviest stones
            x = heapq.heappop(heap)  # Largest stone (negative value)
            y = heapq.heappop(heap)  # Second largest stone (negative value)

            # If they are not equal, push the difference back
            if x != y:
                heapq.heappush(heap, x - y)  # Difference is negative

        # Step 3: Return the remaining stone or 0 if none remain
        return -heap[0] if heap else 0

```

### 2336. Smallest Number in Infinite Set

```python
import heapq

class SmallestInfiniteSet:

    def __init__(self):
        # Set to track numbers that have been removed (popped)
        self.removed = set()

        # Min-heap to store numbers that are explicitly added back
        self.heap = []

        # Tracks the smallest number in the "infinite" sequence that has not been removed
        self.minimum = 1

    def popSmallest(self) -> int:
        # If the heap is not empty, pop the smallest number from the heap
        if self.heap:
            value = heapq.heappop(self.heap)  # Get the smallest number from the heap
            self.removed.add(value)  # Mark it as removed
            return value  # Return the smallest value

        # If the heap is empty, return the current "minimum" value from the infinite sequence
        value = self.minimum
        self.minimum += 1  # Move to the next smallest infinite number
        self.removed.add(value)  # Mark the current number as removed
        return value

    def addBack(self, num: int) -> None:
        # Only add the number back if:
        # 1. It is smaller than the current "minimum" (i.e., part of the finite sequence).
        # 2. It has been previously removed (i.e., exists in the `removed` set).
        if num < self.minimum and num in self.removed:
            heapq.heappush(self.heap, num)  # Add the number back to the heap
            self.removed.remove(num)  # Remove it from the `removed` set since it's now available again

```

### 2530. Maximal Score After Applying K Operations

```python
import heapq
import math
from typing import List

class Solution:
    def maxKelements(self, nums: List[int], k: int) -> int:
        # Step 1: Convert nums to a max-heap by storing negative values
        heap = [-num for num in nums]
        heapq.heapify(heap)  # O(n) to build the heap

        score = 0

        # Step 2: Perform k operations
        for _ in range(k):
            value = -heapq.heappop(heap)  # Get the maximum value
            score += value
            # Push back the ceiling of value / 3 as a negative value
            heapq.heappush(heap, -math.ceil(value / 3))

        return score

```

### 3296. Minimum Number of Seconds to Make Mountain Height Zero

```python
import heapq
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        # Initialize a min-heap with (time to complete first task, worker's original time, task count)
        heap = [(time, time, 1) for time in workerTimes]
        heapq.heapify(heap)

        total_time = 0

        while mountainHeight > 0:
            # Get the worker who finishes the next task the earliest
            cur_time, base_time, count = heapq.heappop(heap)

            # Update the total time to reflect the last finished task
            total_time = max(total_time, cur_time)

            # Decrease the remaining mountain height (assign one task to this worker)
            mountainHeight -= 1

            # Push the worker back into the heap with their updated next task time
            next_time = cur_time + base_time * (count + 1)
            heapq.heappush(heap, (next_time, base_time, count + 1))

        return total_time

```

### 1801. Number of Orders in the Backlog

```python
import heapq
from typing import List

class Solution:
    def getNumberOfBacklogOrders(self, orders: List[List[int]]) -> int:
        MOD = 10**9 + 7

        # Heaps for buy and sell orders
        buy_heap = []  # Max-heap for buy orders (negative prices for max-heap simulation)
        sell_heap = []  # Min-heap for sell orders (positive prices for natural min-heap behavior)

        # Process each order
        for price, amount, orderType in orders:
            if orderType == 0:  # Buy order
                # Try to match with the lowest-priced sell orders
                while sell_heap and sell_heap[0][0] <= price and amount > 0:
                    sell_price, sell_amount = heapq.heappop(sell_heap)  # Get the lowest sell price
                    if sell_amount <= amount:
                        # Sell order is fully matched
                        amount -= sell_amount
                    else:
                        # Partially match the sell order
                        heapq.heappush(sell_heap, (sell_price, sell_amount - amount))
                        amount = 0  # Buy order is fully satisfied
                # If there is any remaining amount in the buy order, add it to the buy heap
                if amount > 0:
                    heapq.heappush(buy_heap, (-price, amount))

            else:  # Sell order
                # Try to match with the highest-priced buy orders
                while buy_heap and -buy_heap[0][0] >= price and amount > 0:
                    buy_price, buy_amount = heapq.heappop(buy_heap)  # Get the highest buy price
                    if buy_amount <= amount:
                        # Buy order is fully matched
                        amount -= buy_amount
                    else:
                        # Partially match the buy order
                        heapq.heappush(buy_heap, (buy_price, buy_amount - amount))
                        amount = 0  # Sell order is fully satisfied
                # If there is any remaining amount in the sell order, add it to the sell heap
                if amount > 0:
                    heapq.heappush(sell_heap, (price, amount))

        # Calculate the total remaining orders in the backlog
        backlog = 0
        for _, amount in buy_heap:
            backlog += amount  # Sum up remaining buy orders
        for _, amount in sell_heap:
            backlog += amount  # Sum up remaining sell orders

        # Return the total backlog modulo 10^9 + 7
        return backlog % MOD

```

### 2233. Maximum Product After K Increments

```python
import heapq
from typing import List

class Solution:
    def maximumProduct(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7

        # Step 1: Convert nums into a min-heap
        heapq.heapify(nums)

        # Step 2: Increment the smallest element k times
        for _ in range(k):
            heapq.heappush(nums, heapq.heappop(nums) + 1)

        # Step 3: Calculate the product of all elements, taking modulo at each step
        result = 1
        for num in nums:
            result = (result * num) % MOD

        return result

```

### 1834. Single-Threaded CPU

```python
import heapq
from typing import List

class Solution:
    def getOrder(self, tasks: List[List[int]]) -> List[int]:
        # Step 1: Add original indices to tasks and sort by enqueue time
        indexed_tasks = [(et, pt, i) for i, (et, pt) in enumerate(tasks)]
        indexed_tasks.sort(key=lambda x: x[0])  # Sort by enqueueTime

        # Step 2: Use a min-heap for processing tasks
        heap = []
        time = 0
        result = []
        i = 0  # Pointer for traversing indexed_tasks

        # Step 3: Simulate CPU processing
        while len(result) < len(tasks):
            # Add all tasks that are available at the current time
            while i < len(indexed_tasks) and indexed_tasks[i][0] <= time:
                heapq.heappush(heap, (indexed_tasks[i][1], indexed_tasks[i][2]))  # (processingTime, index)
                i += 1

            if heap:
                # Process the next task
                processing_time, index = heapq.heappop(heap)
                time += processing_time
                result.append(index)
            else:
                # If no tasks are available, move time to the next task's enqueueTime
                time = indexed_tasks[i][0]

        return result

```