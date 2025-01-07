## Two Pointers

**Key Idea**
- Use two pointers to scan or process elements in a linear fashion, reducing the need for nested loops and improving performance.
- The pointers can move in the same direction, opposite directions, or in other specific patterns depending on the problem.


### 27. Remove Element

```python
from typing import List

class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        # The slow pointer keeps track of the position where the next valid (non-val) element should be placed.
        slow = 0

        # Iterate through the array with the fast pointer
        for fast in range(len(nums)):
            # Check if the current element is not equal to the target value
            if nums[fast] != val:
                # If the condition is met, copy the element at the fast pointer
                # to the position of the slow pointer
                nums[slow] = nums[fast]
                # Increment the slow pointer to prepare for the next valid element
                slow += 1

        # Return the length of the modified array (number of elements that are not val)
        return slow

```

### 26. Remove Duplicates from Sorted Array

```python
from typing import List

from typing import List

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        # Initialize the slow pointer to track the position of unique elements
        slow = 0

        # Start iterating with the fast pointer from the second element (index 1)
        for fast in range(1, len(nums)):
            # Check if the current element is different from the previous one
            if nums[fast] != nums[fast - 1]:
                # Increment the slow pointer to prepare for the new unique element
                slow += 1
                # Copy the unique element to the position tracked by the slow pointer
                nums[slow] = nums[fast]

        # Return the new length of the array with unique elements
        return slow + 1

```

### 283. Move Zeros

```python
from typing import List

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        Moves all zeros in the list to the end while maintaining the relative order of non-zero elements.
        """
        # The slow pointer keeps track of where the next non-zero element should be placed
        slow = 0

        # The fast pointer iterates through the array to find non-zero elements
        for fast in range(len(nums)):
            # Check if the current element is non-zero
            if nums[fast] != 0:
                # Swap the non-zero element at the fast pointer with the element at the slow pointer
                nums[slow], nums[fast] = nums[fast], nums[slow]
                # Increment the slow pointer to prepare for the next non-zero element
                slow += 1

```

### 844. Backspace String Compare

```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        # Convert the input strings to lists for mutability
        s, t = list(s), list(t)

        # Initialize two pointers to track the processed portion of each string
        slow_s, slow_t = 0, 0

        # Process the first string 's'
        for fast in range(len(s)):
            if s[fast] != '#':
                # If the character is not a backspace, keep it in the result
                s[slow_s] = s[fast]
                slow_s += 1
            else:
                # If it is a backspace, remove the last valid character (if any)
                if slow_s > 0:
                    slow_s -= 1

        # Process the second string 't'
        for fast in range(len(t)):
            if t[fast] != '#':
                # If the character is not a backspace, keep it in the result
                t[slow_t] = t[fast]
                slow_t += 1
            else:
                # If it is a backspace, remove the last valid character (if any)
                if slow_t > 0:
                    slow_t -= 1

        # If the lengths of the processed strings differ, they cannot be equal
        if slow_s != slow_t:
            return False

        # Compare the processed portions of both strings
        return s[:slow_s] == t[:slow_t]


class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        def getNextValidIndex(string: str, index: int) -> int:
            count = 0  # Counter for backspaces
            while index >= 0:
                if string[index] == '#':  # Encounter a backspace
                    count += 1
                elif count > 0:  # Skip a valid character if there's a backspace to apply
                    count -= 1
                else:  # Valid character found
                    return index
                index -= 1  # Move to the previous character
            return -1  # Return -1 if no valid character is found

        # Initialize pointers for both strings starting at the last index
        i, j = len(s) - 1, len(t) - 1

        while i >= 0 or j >= 0:
            # Get the next valid character indices for both strings
            i = getNextValidIndex(s, i)
            j = getNextValidIndex(t, j)

            # Compare the characters at the current valid indices
            if i >= 0 and j >= 0 and s[i] != t[j]:
                return False  # Characters differ, so strings are not equal

            # Check if one string is exhausted while the other still has valid characters
            if (i >= 0) != (j >= 0):
                return False  # Unequal lengths after processing backspaces

            # Move pointers to the previous character
            i -= 1
            j -= 1

        return True  # Strings are equal after applying backspaces


class Solution:
    """
    Stack-Based Solution
    """
    def backspaceCompare(self, s: str, t: str) -> bool:
        def build(string):
            stack = []
            for char in string:
                if char != '#':
                    stack.append(char)
                elif stack:
                    stack.pop()
            return stack
        
        return build(s) == build(t)


```

### 151. Reverse Words in a String

The code is about how to remove duplicated spaces by using 2 pointers.

```python
from typing import List

def removeDuplicatedSpaces(s: List[str]) -> str:
    slow, fast = 0, 0  # Initialize two pointers, slow for the resulting string and fast for traversal.
    
    while fast < len(s):  # Iterate through the input character list.
        if s[fast] != ' ':  # If the current character is not a space:
            if slow > 0:  # If slow > 0, it means this is not the first word.
                s[slow] = ' '  # Insert a single space before adding the next word.
                slow += 1
            
            # Copy the word (non-space characters) to the slow pointer position.
            while fast < len(s) and s[fast] != ' ':
                s[slow] = s[fast]
                slow += 1
                fast += 1
        fast += 1  # Skip extra spaces after processing a word.

    # Join the valid portion of the list into a string and return.
    return ''.join(s[:slow])

```

### 206. Reverse Linked List

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None  # If the list is empty, return None
        
        # Initialize pointers
        prev = None  # Previous node (initially None)
        current = head  # Current node (starting at the head)
        
        # Traverse the list
        while current:
            # Save the next node
            next_node = current.next
            
            # Reverse the link
            current.next = prev
            
            # Move the pointers forward
            prev = current
            current = next_node
        
        # At the end, prev points to the new head
        return prev

```

### 24. Swap Nodes in Pairs

```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None  # Return None if the list is empty
        
        # Create a dummy node to simplify edge cases (e.g., swapping the head node)
        dummy = ListNode(0, head)
        
        # Initialize pointers
        prev = dummy  # Tracks the node before the current pair
        current = head  # Tracks the first node of the current pair

        # Traverse the list in pairs
        while current and current.next:
            # Identify the nodes to be swapped
            then = current.next  # The second node of the pair

            # Perform the swap
            current.next = then.next  # Link the first node to the node after the pair
            then.next = current  # Link the second node to the first node
            prev.next = then  # Link the previous node to the second node

            # Move pointers forward to the next pair
            prev = current  # Move prev to the end of the current pair
            current = current.next  # Move current to the next pair's first node

        # Return the new head of the list
        return dummy.next

```

### 19. Remove Nth Node From End of List

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        if not head:
            return None  # If the list is empty, return None

        # Create a dummy node pointing to the head of the list
        dummy = ListNode(0, head)
        
        # Initialize two pointers starting from the dummy node
        slow, fast = dummy, dummy

        # Move the fast pointer n steps ahead
        for _ in range(n):
            fast = fast.next
        
        # Move both pointers until fast reaches the end of the list
        # At this point, slow will point to the node just before the target node
        while fast and fast.next:
            slow = slow.next
            fast = fast.next

        # Remove the nth node from the end
        slow.next = slow.next.next

        # Return the head of the updated list (dummy.next skips the dummy node)
        return dummy.next

```

### 15. 3Sum

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()  # Sort the array to facilitate the two-pointer approach
        result = []  # Store the resulting triplets
        
        for i in range(len(nums)-2):
            # If the current number is greater than 0, break (no further triplets can sum to 0)
            if nums[i] > 0:
                break

            # Skip duplicate elements to avoid duplicate triplets
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            # Define the target complement to find using two pointers
            complement = -nums[i]
            j, k = i + 1, len(nums) - 1  # Two pointers: start and end of the remaining array
            
            while j < k:
                total = nums[j] + nums[k]  # Sum of the two-pointer elements

                if total < complement:
                    j += 1  # Move the left pointer right to increase the sum
                elif total > complement:
                    k -= 1  # Move the right pointer left to decrease the sum
                else:
                    # Triplet found
                    result.append([nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1

                    # Skip duplicates for the second element of the triplet
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1

                    # Skip duplicates for the third element of the triplet
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1

        return result  # Return the list of unique triplets

```

### 18. 4Sum

```python
from typing import List

class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()  # Sort the array to facilitate the two-pointer approach
        result = []  # Store the resulting quadruplets

        # Outer loop: Fix the first number
        for i in range(len(nums) - 3):
            # Skip duplicates for the first number
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            # Second loop: Fix the second number
            for j in range(i + 1, len(nums) - 2):
                # Skip duplicates for the second number
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue

                # Two-pointer approach for the remaining two numbers
                complement = target - nums[i] - nums[j]
                l, r = j + 1, len(nums) - 1  # Initialize left and right pointers

                while l < r:
                    total = nums[l] + nums[r]

                    if total < complement:
                        l += 1  # Move the left pointer to increase the sum
                    elif total > complement:
                        r -= 1  # Move the right pointer to decrease the sum
                    else:
                        # Quadruplet found
                        result.append([nums[i], nums[j], nums[l], nums[r]])
                        l += 1
                        r -= 1

                        # Skip duplicates for the third number
                        while l < r and nums[l] == nums[l - 1]:
                            l += 1

                        # Skip duplicates for the fourth number
                        while l < r and nums[r] == nums[r + 1]:
                            r -= 1

        return result


from typing import List

class Solution:
    """
    Finds all unique combinations of k numbers in the array that sum to the target.
    """
    def kSum(self, nums: List[int], target: int, k: int) -> List[List[int]]:
        def twoSum(nums: List[int], target: int) -> List[List[int]]:
            result = []
            left, right = 0, len(nums) - 1  # Two pointers
            
            while left < right:
                total = nums[left] + nums[right]

                if total < target:
                    left += 1  # Increase the sum by moving the left pointer right
                elif total > target:
                    right -= 1  # Decrease the sum by moving the right pointer left
                else:
                    # Valid pair found
                    result.append([nums[left], nums[right]])
                    left += 1
                    right -= 1

                    # Skip duplicate elements
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1

            return result

        # Base case: if nums is empty, return no results
        if not nums:
            return []

        # Optimization: Early termination if the target is impossible
        avg = target // k
        if nums[0] > avg or nums[-1] < avg:
            return []

        # Base case for recursion: Use twoSum for k == 2
        if k == 2:
            return twoSum(nums, target)

        # Recursive case for k > 2
        result = []
        for i in range(len(nums)):
            # Skip duplicates for the current element
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            # Recursive call to find (k-1)-sum for the remaining elements
            for subset in self.kSum(nums[i + 1:], target - nums[i], k - 1):
                result.append([nums[i]] + subset)

        return result

```

### 2105. Watering Plants II

```python
class Solution:
    def minimumRefill(self, plants: List[int], capacityA: int, capacityB: int) -> int:
        # Initialize pointers for Alice and Bob
        left, right = 0, len(plants) - 1

        # Initialize current capacities of Alice and Bob's cans
        cur_A, cur_B = capacityA, capacityB

        # Initialize refill count
        refill = 0

        # Traverse the plants array from both ends
        while left <= right:
            if left == right:  # When both Alice and Bob meet at the same plant
                if cur_A < plants[left] and cur_B < plants[right]:
                    refill += 1  # If neither can water the plant, one needs a refill
                break

            # Alice waters the plant on the left
            if plants[left] > cur_A:
                refill += 1  # Refill Alice's can if needed
                cur_A = capacityA
            cur_A -= plants[left]  # Water the plant
            left += 1  # Move Alice to the next plant

            # Bob waters the plant on the right
            if plants[right] > cur_B:
                refill += 1  # Refill Bob's can if needed
                cur_B = capacityB
            cur_B -= plants[right]  # Water the plant
            right -= 1  # Move Bob to the next plant

        return refill

```


### 3132. Find the Integer Added to Array II

1. **Sorting Simplifies Comparison**:
   - Sorting both arrays allows us to align `nums1` and `nums2` element-by-element after adding `x`.

2. **Iterate from Largest Possible `nums1` Values**:
   - To minimize `x`, start with the largest elements of `nums1` and calculate the difference between the smallest element of `nums2` and the chosen element from `nums1`.

3. **Greedy Matching**:
   - Use the calculated `x` to check if adding it to elements in `nums1` can align them with `nums2`.

---

```python
from typing import List

class Solution:
    def minimumAddedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        nums1.sort()  # Sort nums1 for easier comparison
        nums2.sort()  # Sort nums2 to align both arrays

        # Iterate through nums1 in reverse to test with the largest possible numbers
        for i in range(2, -1, -1):  # nums1 has at least 3 elements
            # Calculate the difference between the first element of nums2 and nums1[i]
            x = nums2[0] - nums1[i]

            # Try to match nums1 to nums2 by adding x
            j = 0  # Pointer for nums2
            for num in nums1[i:]:  # Iterate through the remaining elements of nums1
                if num + x == nums2[j]:  # Check if the addition aligns nums1 with nums2
                    j += 1  # Move to the next element in nums2
                
                # If all elements in nums2 are matched, return x
                if j == len(nums2):
                    return x

        return -1  # Return -1 if no valid x is found

```

### 1750. Minimum Length of String After Deleting Similar Ends

```python
class Solution:
    def minimumLength(self, s: str) -> int:
        # Initialize two pointers at the start and end of the string
        left, right = 0, len(s) - 1

        # Process the string as long as the characters at both ends match
        while left < right and s[left] == s[right]:
            char = s[left]  # Store the matching character

            # Move the left pointer inward, skipping all matching characters
            while left <= right and s[left] == char:
                left += 1

            # Move the right pointer inward, skipping all matching characters
            while left <= right and s[right] == char:
                right -= 1

        # Return the length of the remaining string
        # If the pointers cross, return 0 (empty string); otherwise, calculate the length
        return max(0, right - left + 1)


```

### 1855. Maximum Distance Between a Pair of Values

```python
from typing import List

class Solution:
    def maxDistance(self, nums1: List[int], nums2: List[int]) -> int:
        i, j = 0, 0  # Initialize two pointers for nums1 and nums2
        dist = 0     # Variable to store the maximum distance

        # Traverse both arrays with the two-pointer approach
        while i < len(nums1) and j < len(nums2):
            if nums1[i] <= nums2[j]:  # Check if the condition holds
                dist = max(dist, j - i)  # Update the maximum distance
                j += 1  # Move the pointer j to check for larger distances
            else:
                i += 1  # If nums1[i] > nums2[j], increment i to satisfy the condition

        return dist  # Return the maximum distance found

```