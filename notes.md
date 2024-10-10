# Array

## Binary Search

**Constrains**:
- All integers in `nums` are unique
- `nums` is sorted

**Solution**:

`Olog(n)`
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        start, end = 0, len(nums)  # [start, end)

        # It's important to note that the open/closed nature of the interval determines how we update end and the choose the loop termination condition
        while start < end:
            mid = start + (end - start) // 2
        
            if nums[mid] < target:
                start = mid + 1
            elif nums[mid] > target:
                end = mid
            else:
                return mid
        return -1
        
```

## Remove Element

Input: nums = [0,1,2,2,3,0,4,2], val = 2 (in-place)

Output: 5 (nums = [0,1,4,0,3,\_,\_,\_])

**Solution**

1. Brute Force  `O(n^2)`
 ```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i, k = 0, 0
        l = len(nums)
        remove_count = 0
        
        while i < l - remove_count:
            if nums[i] != val:
                k += 1
                i += 1
            else:
                for j in range(i, l - remove_count - 1):
                    nums[j] = nums[j + 1]
                remove_count += 1
        return k

```

2. Fast and Slow Pointer  `O(n)`
```python
class solution:
    def removeElement(self, nums: List[int], val:int) -> int:
        slow, fast = 0, 0
        l = len(nums)

        while fast < l:
            # Slow is used to collect values that are not equal to val. If the value at fast is not equal to val replace it at slow's position
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1

```

## Squares of a Sorted Array  

**solution strategy**:  `O(n)`

The array is sorted; negative numbers may have larger squares.

Thus, the maximum squares are at the two ends of the array, not in the middle.

We can use the **two-pointer technique**, with `i` at the start and `j` at the end.

We create a new array `result`· of the same size as `A`·, with `k` pointing to the end of `result`.

```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        left, right, end = 0, len(nums) - 1, len(nums) - 1
        ret = [0] * len(nums)

        while end >= 0:
            if nums[left] ** 2 > nums[right] ** 2:
                ret[end] = nums[left] ** 2
                left += 1
            else: 
                ret[end] = nums[right] ** 2
                right -= 1
            end -= 1
        return ret
```

## Minimum Size Subarray Sum

**Input**: target = 7, nums = [2,3,1,2,4,3]

**Output**: 2

**Explanation**: The subarray [4,3] has the minimal length under the problem constraint.

**Solution strategy**: `O(n)`

The **sliding window technique** involves continuously adjusting the starting and ending positions of a subsequence to derive the desired result.

The **sliding window** can also be understood as a type of **two-pointer technique**.

1. **What is inside the window**
   
   The window is defined as the minimum length of a contiguous subarray whose sum is `>= target`.

2. **How to move the starting position of window**

   If the current window's sum is greater than or equal to `target`, the starting position of window should move forward (i.e., the window should be shrunk).

3. **How to move the ending position of the window**

   The ending position of window is represented by the pointer that traverses the array, which corresponds to the index in the `for` loop. 


```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        start, end = 0, 0
        size = len(nums)
        min_len = float('inf')
        cur_sum = 0

        while end < size:
            cur_sum += nums[end]

            while cur_sum >= target:
                min_len = min(min_len, end - start + 1)
                cur_sum -= nums[start]
                start += 1
            end += 1
        return min_len if min_len != float('inf') else 0 

```

## Spiral Matrix II

`O(n)`

Simulate the process of drawing a matrix in a clockwise direction:
1. Fill the top row from left to right
2. Fill the right column form top to bottom
3. Fill the bottom row from right to left
4. Fill the left column form bottom to top
Continue this process in concentric circles form the outside in. For each of the four sides, we need to determine how to draw them, ensuring that each side adheres to **consistent left-closed, right-open interval**.

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0] * n for _ in range(n)]
        loop, mid = n // 2, n // 2
        row, col = 0, 0
        num = 1

        for offset in range(1, loop + 1):
            # from left to right
            for i in range(col, n - offset):
                matrix[row][i] = num
                num += 1
            # from top to bottom
            for i in range(row, n - offset):
                matrix[i][n - offset] = num
                num += 1
            # from right to left
            for i in range(n - offset, col, -1):
                matrix[n - offset][i] = num
                num += 1
            # from bottom to top
            for i in range(n - offset, row, -1):
                matrix[i][col] = num
                num += 1
            row += 1
            col += 1
        
        if n % 2 != 0:
            matrix[mid][mid] = num
        
        return matrix

```

## Interval Sum

**The prefix sum** is very useful when it comes to calculating range sums.

```python
class Solution:
    def intervalSum(self, nums: List[int], intervals: List[tuple]) -> List[int]:
        pre_sum = []
        sum = 0

        for i in range(len(nums)):
            sum += nums[i]
            pre_sum.append(sum)
        
        results = []
        for start, end in intervals:
            if start == 0:
                result.append(pre_sum[end])
            else:
                result.append(pre_sum[end] - pre_sum[start - 1])
        return result
```

# Linked List

## Remove Elements
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        cur = head
        pre = None
        while cur != None:
            if cur.val == val:
                if cur == head:  ## Head node
                    pre = cur
                    cur = cur.next
                    head = cur
                    del pre
                    pre = None
                elif cur.next == None:  # Tail Node
                    pre.next = None
                    del cur
                    cur = None
                else:
                    pre.next = cur.next
                    temp = cur
                    cur = cur.next
                    del temp
                    temp = None
            else:
                pre = cur
                cur = cur.next
        return head
```

A more streamlined version of the code.
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        # A dummy head node is created to simplify handling of edge cases, particularly when removing nodes at the beginning of the list
        dummy_head = ListNode(next=head)

        cur = dummy_head
        while cur.next != None:
            if cur.next.val == val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return dummy_head.next
```
 
 ## Design Linked List

 ```python
 class ListNode:
    def __init__(self, val=val, next=None):
        self.val = val
        self.next = None

class MyLinkedList:
    def __init__(self):
        self.dummy_head = ListNode()
        self.size = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1

        cur = self.dummy_head.next
        for _ in range(index):
            cur = cur.next
        return cur.val
        
    def addAtHead(self, val: int) -> None:
        self.dummy_head.next = ListNode(val=val, next=self.dummy_head.next)
        self.size += 1
        
    def addAtTail(self, val: int) -> None:
        cur = self.dummy_head
        while cur.next != None:
            cur = cur.next
        cur.next = ListNode(val=val, next=None)
        self.size += 1
        
    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:  ## Cannot be index >= size, it doesn't allow adding a new node when the index is equal to the size of the list 
            return

        cur = self.dummy_head
        for _ in range(index):
            cur = cur.next
        cur.next = ListNode(val=val, next=cur.next)
        self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return

        cur = self.dummy_head
        for _ in range(index):
            cur = cur.next
        cur.next = cur.next.next
        self.size -= 1 
 ```

 ## Reverse Linked List

 ```python
 class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre = None
        cur = head

        while cur is not None:
            nex = cur.next  # Save the next node
            cur.next = pre  # Reverse the current node's pointer
            pre = cur       # Move `pre` to the current node
            cur = nex       # Move to the next node
        
        return pre  # pre will be the new head of the reversed list
 ```

 ## Swap Nodes in Pairs

 ```python
#  class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # Create a dummy node to make edge cases easier (e.g., list has only one element)
        dummy_head = ListNode(val=0, next=head)
        pre = dummy_head
        cur = head

        # Loop through the list while there are pairs to swap
        while cur and cur.next:
            nex = cur.next
            
            # Perform the swap
            pre.next = nex
            cur.next = nex.next
            nex.next = cur
            
            # Move the pointers forward for the next pair
            pre = cur
            cur = cur.next

        # Return the next node of dummy_head, which is the new head of the list
        return dummy_head.next
 ```

 ## Remove Nth Node From End of List

 **Solution**: `O(n)`

1. **Use Two Pointers**: Solve this problem in one pass using two pointers, a fast and a slow pointer, known as the "two-pointer technique".

2. **Move the Fast Pointer**: Initially, advance the fast pointer by n nodes to create a gap between the fast and the slow pointers. This gap helps in identifying the nth node from the end as when the fast pointer reaches the end of the list `(None)`, the slow pointer will be just before the nth node from the end.

3. **Move Both Pointers Together**: Once the fast pointer is n nodes ahead, move both pointers simultaneously until the fast pointer reaches the end of the list. At this point, the slow pointer will be positioned right before the node to be removed.

4. **Remove the Node**: Modify the next pointer of the slow pointer to skip the node that needs to be removed by setting `slow.next` to `slow.next.next`. This effectively removes the desired node from the list.

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy_head = ListNode(val=0, next=head)
        slow = fast = dummy_head

        for _ in range(n + 1):
            fast = fast.next

        while fast:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next

        return dummy_head.next
```

## Intersection of Two Linked List
`Time: O(m+n); Space: O(1)`

1. 
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        curA = headA
        curB = headB

        la = self.getLength(headA)
        lb = self.getLength(headB)

        if la > lb:
            for _ in range(la - lb):
                curA = curA.next
        
        if lb > la:
            for _ in range(lb - la):
                curB = curB.next

        while curA is not curB:
            curA = curA.next
            curB = curB.next

        return curA

    def getLength(self, head: ListNode) -> int:
        l = 0
        while head:
            l += 1
            head = head.next
        return l
```
2. 
```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        curA = headA
        curB = headB

        while curA is not curB:
            curA = curA.next if curA else headB
            curB = curB.next if curB else headA

        return curA
```

## Linked List Cycle II

**Solution**: 

**Floyd's Cycle Detection Algorithm**

1. **Two Pointers at Different Speeds**: Use two pointers, slow and fast. The slow pointer moves one step at a time, while the fast pointer moves two steps at a time.
2. **Detection Phase**: If there is a cycle, the fast pointer will eventually meet the slow pointer within the cycle.
3. **Finding the Start of the Cycle**: Once a cycle is detected (when slow meets fast), move one pointer to the head of the list and keep the other at the meeting point. Then move both pointers one step at a time. The point where they meet again is the start of the cycle. 

Explanation **Finding the Start of the Cycle**:

Assume:

- The distance from the head of the list to the start of the cycle is $a$ nodes.
- The distance from the start of the cycle to the point where the slow and fast pointers first meet is $b$ nodes.
- The length of the cycle is $c$ nodes.

When the slow and fast pointers first meet within the cycle:

- The slow pointer has traveled $a+b$ steps.
- The fast pointer has traveled $a+b+k⋅c$ steps (where $k$ is the number of times the hare has looped around the cycle).

Since the fast pointer travels twice as fast as the slow pointer, we have: $2(a+b)=a+b+k⋅c$ => $a+b=k⋅c$.

```python
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return None

        slow = fast = head
        has_cycle = False

        # Detection phase
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                has_cycle = True
                break

        # Finding the start of the cycle
        if has_cycle:
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow

        return None
```

# Hash Table

A **hash table** is a data structure that stores key-value pairs, allowing fast data retrieval. It uses a **hash function** to convert keys into array indices, where values are stored. Hash tables offer average **O(1)** time complexity for search, insert, and delete operations. To handle collisions (when two keys hash to the same index), common techniques include **chaining** (using linked lists) and **open addressing** (finding another open slot).

## Valid Anagram

Given two string `s` and `t` return `true` if `t` is an anagram of `s`, and `false` otherwise.

**Solution**:
```python
lass Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        s_dict = {}
        t_dict = {}

        for char in s:
            s_dict[char] = s_dict.get(char, 0) + 1
        
        for char in t:
            t_dict[char] = t_dict.get(char, 0) + 1

        return s_dict == t_dict

```
```python
## If char exists in the dictionary, get(char) returns the current count of that character.
## If char does not exist in the dictionary, get(char, 0) returns 0, which serves as a default value indicating the character hasn't been seen yet.
s_dict[char] = s_dict.get(char, 0) + 1
```

**Alternative (Using Python's collections.Counter)**:
```python
from collections import Counter

def isAnagram(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)
```

## Intersection of Two Arrays

**Solution**: (Hash Table)
```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1_dict = {}
        inter_list = []

        for num in nums1:
            nums1_dict[num] = 1

        for num in nums2:
            if nums1_dict.get(num, 0):
                nums1_dict[num] += 1

        for key in nums1_dict:
            if nums1_dict[key] > 1:
                inter_list.append(key)

        return inter_list

```

Using `set`
```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Convert nums1 and nums2 to sets to remove duplicates and optimize membership check
        set1 = set(nums1)
        set2 = set(nums2)
        
        # Use set intersection to find common elements
        return list(set1.intersection(set2))
```

## Happy Number

**Solution**:

```python
class Solution:
    def digitsSum(self, n: int) -> int:
        sum = 0
        while n > 0:
            sum += (n % 10) ** 2  # Add the square of the last digit
            n = n // 10  # Remove the last digit
        return sum
    
    # Using Has Table
    def isHappy(self, n: int) -> bool:
        sum_dict = {}
        while n != 1:
            n = self.digitsSum(n)
            if n in sum_dict:
                return False  # Cycle detected
            sum_dict[n] = 1  # Mark this sum as seen
        return True
    
    # Using Fast-slow pointers
    def isHappy(self, n: int) -> bool:
        slow = n
        fast = self.digitsSum(n)
        
        # Continue until fast reaches 1 (happy number) or slow catches up with fast (cycle detected)
        while fast != 1 and slow != fast:
            slow = self.digitsSum(slow)          # Move slow one step
            fast = self.digitsSum(self.digitsSum(fast))  # Move fast two steps
            
        # If fast reaches 1, it's a happy number
        return fast == 1
```

## Two Sum

**Solution**:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_to_index = {}

        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_to_index:
                return [num_to_index[complement], i]

            num_to_index[num] = i
        return []

```

## Four Sum Count

**Solution**:

```python
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        sum_12_dict = {}
        for num1 in nums1:
            for num2 in nums2:
                sum_12_dict[num1 + num2] = sum_12_dict.get(num1 + num2, 0) + 1

        count = 0
        for num3 in nums3:
            for num4 in nums4:
                complement = -(num3 + num4)
                if complement in sum_12_dict:
                    count += sum_12_dict[complement]

        return count

```

## Ransom Note

**Solution**:
```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        magazine_dict = {}

        for char in magazine:
            magazine_dict[char] = magazine_dict.get(char, 0) + 1

        for char in ransomNote:
            if char not in magazine_dict or magazine_dict[char] == 0:
                return False
            magazine_dict[char] -= 1
            
        return True

```

## 3Sum

**Solution**:

Steps:
1. **Sort the array**: Sorting helps in using two-pointer technique and avoiding duplicates easily.
2. **Iterate through the array**: For each number, treat it as a target `nums[i]`, and use two pointers (one starting from `i + 1` and another from the end of the array) to find two other numbers that sum up to `-nums[i]`.
3. **Avoid duplicates**: After finding a valid triplet, skip any duplicate numbers to avoid redundant result.

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result = []
        size = len(nums)

        for i in range(size - 2):
            # Since the array is sorted, once nums[i] > 0,
            # there is no way to get a sum of zero because all numbers after it will be positive.
            if nums[i] > 0:
                return result
            # We don't want to process the same number multiple times
            # for the same position `i` to avoid adding duplicate triplets in the result.
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            left, right = i + 1, size - 1
            while left < right:
                total = nums[i] + nums[left] + nums[right]

                if total == 0:
                    result.append([nums[i], nums[left], nums[right]])

                    # [-1, 0, 0, 0, 0, 1, 1, 1, 1]
                    # Move the `left` pointer to the right to avoid duplicates.
                    # Skip any consecutive duplicates of nums[left].
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    # Move the `right` pointer to the left to avoid duplicates.
                    # Skip any consecutive duplicates of nums[right].
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1

                    left += 1
                    right -= 1
                elif total < 0:
                    left += 1
                else:
                    right -= 1

        return result

```

`if i > 0 and nums[i] == nums[i - 1]` 

In the case of `[-1, -1, 0, 1, 1, 2]`, when `i = 0`, the search area is from `i + 1` to `len(nums) - 1`, which corresponds to the subarray `[-1, 0, 1, 1, 2]`. This search area includes all possible triplet results where `-1` is the first number. When `i = 1` (where `nums[i] = -1` again), the search area becomes `[0, 1, 1, 2]`, which is a subset of the previous search area (because the first `-1` has already been processed). Without the condition `if i > 0 and nums[i] == nums[i - 1]`, this would lead to duplicate triplet results. By skipping duplicate values of `nums[i]`, we avoid redundant calculations and ensure each unique triplet is added to the result only once.


## 4Sum

**Solution**:
```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        result = []
        size = len(nums)

        for k in range(size - 3):
            if nums[k] > target and target > 0:
                return result
            if k > 0 and nums[k] == nums[k - 1]:
                continue
            
            for i in range(k + 1, size - 2):
                sum_k_i = nums[k] + nums[i]
                if sum_k_i > target and target > 0:
                    break
                if i > k + 1 and nums[i] == nums[i - 1]:
                    continue
                
                left, right = i + 1, size - 1
                complement = target - sum_k_i
                while left < right:
                    sum_left_right = nums[left] + nums[right]
                    if sum_left_right == complement:
                        result.append([nums[k], nums[i], nums[left], nums[right]])

                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        left += 1
                        right -= 1
                    elif sum_left_right < complement:
                        left += 1
                    else:
                        right -= 1
        return result
```

**Caution**: 
```python
if sum_k_i > target and target > 0:
    break  # break not return
```
Case: [-2, 0, 0, 1, 3, 7], target = 4 => `-2 + 7 > 4` if return `[0, 0, 1, 3]` will be ignored.

# String

## Remove Duplicated Spaces
**Input**: " This    is an example   "

**Output**: "This is an example"

**Solution**:
```python
class Solution:
    def removeDuplicatedSpaces(self, s:str) -> str:
        s = list(s)
        # 'write' is the position where we write the cleaned-up characters,
        # 'read' is the position we are currently reading from.
        write, read = 0, 0

        while read < len(s):
             # If the current character is not a space
            if s[read] != ' ':
                # If write != 0, we are not at the beginning, so we place a space before the next word
                if write != 0:
                    s[write] = ' '
                    write += 1
                
                while read < len(s) and s[read] != ' ':
                    s[write] = s[read]
                    read += 1
                    write += 1

            # Move the read pointer to the next character (skip over spaces)
            read += 1
        return ''.join(s[:write])

```

### KMP
A highly efficient method used to search for a pattern within a text **without backtracking the text pointer**, completing the task in linear time `O(n + m)`, where `n` is the length of the text and `m` is the length of the pattern. 

Pattern|A|B|A|B|A|A|
|-|-|-|-|-|-|-|
Index|0|1|2|3|4|5
Next |0|0|1|2|3|1

```python
class Solution:
    def getNext(self, pattern:str) -> list:
        m = len(pattern)
        next = [0] * m
        # k represents the length of the current common longest prefix and suffix 
        # and also serves as a the end of prefix
        k = 0

        for i in range(1, m):  # i represents the end of suffix
            # Adjust k to the longest prefix length that also suffixes up to pattern[i-1]
            while k > 0 and pattern[k] != pattern[i]:
                k = next[k-1]

            if pattern[k] == pattern[i]:
                k += 1

            next[i] = k
        return next

    def kmp(self, text:str, pattern:str) -> int:
        next = self.getNext(pattern)
        n = len(text)
        m = len(pattern)

        i, j = 0, 0
        while i < n and j < m:
            if text[i] == pattern[j]:
                i += 1
                j += 1
            elif j == 0:
                # If there's no match and j is 0, move to the next character in the text
                i += 1
            else:
                j = next[j-1]

        return i - j if j == m else -1

```

## Repeated Substring Pattern

Given a string  `s`, check if it can be constructed by taking a substring of it and appending multiple copies of the substring together.

**Solution**: 

1. **Concatenate the string with itself**: When you join s with itself (i.e., `s + s`), all possible shifts of the string appear.
2. **Remove the first and last characters**: This prevents a false match where the original string overlaps.
3. **Check if the original string exists in this modified version**: If s is found in the middle, it means s can be made by repeating a substring.
   
```python
class Solution:
    def repeatedSubstringPattern(self, s:str) -> bool:
        ss = (s + s)[1:-1]
        return s in ss  # KMP
```

# Stack and Queue

In Python, **Stack** and **Queue** are two common data structures that are used for managing elements with specific order properties.

## Stack

**Last In, First Out (LIFO)**: The last element added to the stack is the first one to be removed.

Operations:

- **Push**: Add an element to the top of the stack.
- **Pop**: Remove the element form the top of the stack.

Implementation: **List**

```python
stack = []

# Push elements onto the stack
stack.append(1)
stack.append(2)
stack.append(3)

# Pop elements from the stack
top_elements = stack.pop()  # Removes and returns 3 (last in, first out)
```

## Queue

**First In, First Out (FIFO)**: The first element added to the queue is the first one to be removed.

Operations:

- **Enqueue**: Add an element to the end of the queue.
- **Dequeue**: Remove an element from the front of the queue.

Implementation: 

```python
from collections import deque

queue = deque()

# Enqueue elements into the queue
queue.append(1)
queue.append(2)
queue.append(3)

# Dequeue elements from the queue
front_element = queue.popleft()  # Removes and returns 1 (first in, first out)
```

## Implement Queue using Stack

**Solution**:
```python
class MyQueue:
    def __init__(self):
        self.stack_in = []
        self.stack_out = []

    def push(self, x:int) -> None:
        self.stack_in.append(x)

    def pop(self) -> int:
        if self.empty():
            return None

        if self.stack_out:
            return self.stack_out.pop()
        else:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
            return self.stack_out.pop()

    def peek(self) -> int:
        element = self.pop()
        self.stack_out.append(element)
        return element
    
    def empty(self) -> bool:
        return not (self.stack_in or self.stack_out)
```

## Implement Stack using Queues

**Solution**: Only using one queue

```python
from collections import deque

class MyStack:
    def __init__(self):
        self.que = deque()

    def push(self, x:int) -> None:
        self.que.append(s)

    def pop(self) -> int:
        if self.empty():
            return None

        for i in range(len(self.que) - 1):
            self.que.append(self.que.popleft())

        return self.que.popleft()

    def top(self) -> int:
        if self.empty():
            return None

        return self.que[-1]

    def empty(self) -> bool:
        return not self.que
```

## Valid Parentheses

**Solution**:
1. case 1: ['(', '[', '{', '}', ']', '(', ')']  unmatched opening bracket
2. case 2: ['[', '{', '(', '}']  incorrect bracket type and order
3. case 3: ['(', ')', ')']  extra closing bracket

```python
class Solution:
    def isValid(self, s:str) -> bool:
        stack = []
        mapping = {
            '(': ')',
            '[': ']',
            '{': '}'
        }

        for item in s:
            if item in mapping.keys():
                stack.append(mapping[item])
            elif not stack or stack[-1] != item:
                return False
            else:
                stack.pop()
        return True if not stack else False
```

## Remove All Adjacent Duplicates In String

**Solution**:

```python
class Solution:
    def removeDuplicates(self, s: str) -> str:
        stack = []

        for char in s:
            if not stack:
                stack.append(char)
                continue

            if stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)

        return ''.join(stack)
        
```

## Evaluate Reverse Polish Notation

**Solution**:

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        
        for token in tokens:
            if token.lstrip('-').isdigit():
                stack.append(int(token))
            else:
                a = stack.pop()
                b = stack.pop()

                if token == '+':
                    stack.append(b + a)
                
                if token == '-':
                    stack.append(b - a)

                if token == '*':
                    stack.append(b * a)

                if token == '/':
                    stack.append(int(b / a))
        return stack[-1]

```
p.s. 

1. Use `token.lstrip('-').isdigit()` to check if a token represents a valid integer, including negative numbers.
2. In subtraction and division, the operands must be used in the correct order:
   - For subtraction: `b - a` (i.e., subtract `a` from `b`).
   - For division: `b / a` (i.e., divide `b` by `a`).


## Reverse Polish Notation

**Input**: "3 + ( -2 ) * 5"
**Output**: "3 -2 5 * +"

```python
class Solution:
    def RPN(self, expression:str) -> str:
        # Define operators' precedence, large number meaning high priority
        precedence = {
            '+': 1,
            '-': 1,
            '*': 2, 
            '/': 2
        }
        
        output = []
        operators = [] # A stack to store operands

        tokens = expression.spilt()

        for token in tokens:
            if token.lstrip('-').isdigit():
                output.append(token)
                continue

            if token in precedence:
                while operators and operators[-1] != '(' and precedence[operators[-1]] >= precedence[token]:
                    output.append(operators.pop())
                operators.append(token)
            elif token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()

        while operators:
            output.append(operators.pop())

        return ' '.join(output)
        
```
## Sliding Window maximum

**Solution**:

Input: nums = [1,3,-1,-3,5,3,6,7], k = 3

Output: [3,3,5,5,6,7]

Explanation: 

Window position  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;             Max

---------------  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;           -----

[1  3  -1] -3  5  3  6  7  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;      3

 1 [3  -1  -3] 5  3  6  7  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;      3

 1  3 [-1  -3  5] 3  6  7  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;      5

 1  3  -1 [-3  5  3] 6  7  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;      5

 1  3  -1  -3 [5  3  6] 7  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;      6

 1  3  -1  -3  5 [3  6  7]  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;     7

Step:
- Loop through the array.
  1. Remove elements that are out of the bounds of the current window from the front of the queue
  2. Remove elements from the back of the queue that are smaller than the current element because they are no longer useful
  3. Add the current element's index to the queue
  4. Once the window has reached size `k`, the maximum element (at the front of the queue) is added to the result list

```python
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums:List[int], k:int) -> List[int]:
        queue = deque()
        result = []

        for i in range(len(nums)):
            # Remove elements from the deque that are out of this window's bounds
            if queue and queue[0] < i - k + 1:
                queue.popleft()

            # Remove elements that are smaller than the current number from the back of the deque
            while queue and nums[i] > nums[queue[-1]]:
                queue.pop()

             # Add current element's index to the deque
            queue.append(i)

            # Append the max value of the current window to the result once the first window is full
            if i >= k - 1:
                result.append(nums[queue[0]])
        return result

```

## Top K Frequent Elements

**Solution**: Using `Counter`

time complexity: `O(n + klogn)`

```python
import Counter

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        return [key for key, _ in count.most_common(k)]
```

**`heapq` Overview**:

The heap implemented by `heapq` is a min-heap by default, 
meaning the smallest element is always at the top of the heap.

**Common `heapq` Functions**:
1. `heapq.heappush(heap, item)`:
   - Push `item` onto the heap while maintaining the heap invariant (the smallest element stays at the top).
2. `heapq.heappop(heap)`:
   - Pop and return the smallest element from the heap. The heap is automatically restructured to maintain the min-heap property.


```python
import heapq

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        nums_dict = {}

        # Counting Frequencies
        for num in nums:
            nums_dict[num] = nums_dict.get(num, 0) + 1

        pri_que = []
        for key, freq in nums_dict.items():
            heapq,heappush(pri_que, (freq, key))
            if len(pri_que) > k:
                heapq.heappop(pri_que)

        result = [0] * k
        for i in range(k-1, -1, -1):
            result[i] = heapq.heappop(pri_que)[1]
        return result

```