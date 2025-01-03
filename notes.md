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
        return slow

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

p.s. index of root: k; index of left child: 2k + 1; index of right child: 2k + 2

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

# Binary Tree

## Preorder, Inoreder, PostOrder Traversal

```python
class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def preorederTraversal(self, root:Optional[TreeNode]) -> List[int]:
        # root, left, right
        if root is None:
            return []

        return [root.val] + self.preorderTraversal(root.left) + self.preorederTraversal(root.right)

    def inorederTraversal(self, root:Optional[TreeNode]) -> List[int]:
        # left, root, right
        if root is None:
            return []

        return self.preorderTraversal(root.left) + [root.val] + self.preorederTraversal(root.right)

    def postorederTraversal(self, root:Optional[TreeNode]) -> List[int]:
        # left, right, root
        if root is None:
            return []

        return self.preorderTraversal(root.left) + self.preorederTraversal(root.right) + [root.val]

    def iterativePreorderTraversal(self, root:Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        result = []
        traversal_stack = [root]
        while traversal_stack:
            node = traversal_stack.pop()
            result.append(node.val)

            if node.right:
                traversal_stack.append(node.right)

            if node.left:
                traversal_stack.append(node.left)
        return result

    def iterativeInorderTraversal(self, root:Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        result = []
        traversal_stack = []
        cur = root
        while cur or traversal_stack:
            if cur:
                traversal_stack.append(cur)
                cur = cur.left
            else:
                cur = traversal_stack.pop()
                result.append(cur.val)
                cur = cur.right
        return result
        
    def iterativePostorderTraversal(self, root:Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        result = []
        traversal_stack = [root]
        while traversal_stack:
            node = traversal_stack.pop()
            result.append(node.val)

            if node.left:
                traversal_stack.append(node.right)

            if node.right:
                traversal_stack.append(node.left)
        return result[::-1]  # Flip the final array
```

## Level Order Traversal

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []
        traversal_queue = deque()
        traversal_queue.append(root)

        while traversal_queue:
            level = []
            size = len(traversal_queue)
            for _ in range(size):
                node = traversal_queue.popleft()

                level.append(node.val)

                if node.left:
                    traversal_queue.append(node.left)
                if node.right:
                    traversal_queue.append(node.right)
            result.append(level)
        return result

```

## Maximum/Minimum Depth of Binary Tree

**Solution**:

Recursive Method

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        if not (root.left or root.right):
            return 1
        
        # Replace max() with min() to find the minimum depth
        return max(1 + self.maxDepth(root.left), 1 + self.maxDepth(root.right))

```

Iterative Method: Using Queue

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
       if not root:
        return 0

       depth = 0
       traversal_queue = deque([root])
       while traversal_queue:
        depth += 1
        size = len(traversal_queue)
        for _ in range(size):
            node = traversal_queue.popleft()
            if node.left:
                traversal_queue.append(node.left)
            if node.right:
                traversal_queue.append(node.right)
       return depth


class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        depth = 0
        traversal_queue = deque([root])
        while traversal_queue:
         depth += 1
         size = len(traversal_queue)
         for _ in range(size):
            node = traversal_queue.popleft()
            if not (node.left or node.right):
                return depth

            if node.left:
                traversal_queue.append(node.left)
            if node.right:
                traversal_queue.append(node.right)
        return depth
```

## Invert Binary Tree

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None

        if not (root.left or root.right):
            return root

        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
```

## Symmetric Tree

```python
class Solution:
    def compareNode(self, left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
        # Case 1: One of the nodes is None and the other is not, they are not symmetric
        if not left and right:
            return False
        
        if left and not right:
            return False

        # Case 2: Both nodes are None, they are symmetric
        if not left and not right:
            return True

        # Case 3: The values of the current nodes are different, they are not symmetric
        if left.val != right.val:
            return False

        # Recursively check the "outside" pair (left's left with right's right)
        outside = self.compareNode(left.left, right.right)

        # Recursively check the "inside" pair (left's right with right's left)
        inside = self.compareNode(left.right, right.left)

        # The nodes are symmetric only if both outside and inside comparisons are true
        return outside and inside

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        # An empty tree is symmetric
        if not root:
            return True
        
        return self.compareNode(root.left, root.right)
```

## Count Complete Tree Node

**Solution**:

### Complexity Analysis

**Time Complexity:**
- The left and right depth calculations each run in $( O(d) )$, where $( d )$ is the tree depth $(( O(log n) ))$.
- For a complete binary tree, the method returns directly in $( O(log n) )$.
- If the tree is not complete, the method recurses on both subtrees, leading to a time complexity of $(O((log n)^2))$ due to repeated depth calculations.

**Space Complexity:**
- $( O(log n) )$ due to recursion depth, which corresponds to the height of the tree.

```python
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        left_depth, right_depth = 0, 0
        left, right = root.left, root.right

        while left:
            left = left.left
            left_depth += 1
        while right:
            right = right.right
            right_depth += 1

        if left_depth == right_depth:
            return 2 ** (left_depth + 1) - 1

        return self.countNodes(root.left) + self.countNode(root.right) + 1

```

## Balanced Binary Tree

### Ideas to Solve the Problem

To efficiently determine if a binary tree is balanced, we can use a single recursive function that combines depth calculation and balance checking. 
This function will traverse the tree once, checking if each subtree is balanced while calculating its depth. By returning both the balance status and the depth in one traversal, 
we avoid the need for repeated depth calculations, leading to a more efficient solution.

```python
class Solution:
    def check_balance_and_depth(self, root: Optional[TreeNode]) -> (bool, int):
        if not root:
            return True, 0

        left_balanced, left_depth = self.check_balance_and_depth(root.left)
        right_balanced, right_depth = self.check_balance_and_depth(root.right)

        balanced = left_balanced and right_balanced and abs(left_depth - right_depth) <= 1
        return balanced, max(left_depth, right_depth) + 1



    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        return self.check_balance_and_depth(root)[0]
```

## Binary Tree Paths

**Solution**:

1. **Recursive Traversal:** 
   Use a recursive function to traverse the binary tree, starting from the root. At each node, keep track of the current path by adding the node's value to a list.

2. **Path Construction:** 
   When a leaf node (a node with no left or right children) is reached, convert the accumulated path into a string using `"->"` as the separator and store it in the result list.

3. **Backtracking:** 
   After exploring a path through a node, backtrack by removing the last added node from the path. This ensures that the path list is correctly restored, allowing other recursive calls to explore alternative paths.

```python
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        def traversal(root: Optional[TreeNode], path: List[int], result: List[str]):
            path.append(root.val)

            if not (root.left or root.right):
                return result.append("->".join(map(str, path)))

            if root.left:
                traversal(root.left, path, result)
                # Backtracking: remove the last node to explore other paths
                path.pop()
            if root.right:
                traversal(root.right, path, result)
                # Backtracking: remove the last node to explore other paths
                path.pop()
        
        
        path = []
        result = []
        traversal(root, path, result)
        return result

```
## Sum of Left Leaves

```python
class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        left_sum = 0

         # Check if the left node exists and is a leaf
        if root.left and not (root.left.left or root.left.right):
            left_sum += root.left.val
            
         # Recursively sum left leaves in both left and right subtrees
        return left_sum + self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)

```

## Find Bottom Left Tree Value

```python
class Solution:
    def findBottomLeftTreeValue(self, root: Optional[TreeNode]) -> int:
        if not root:
            return None

        traversal_queue = deque([root])
        left_bottom_value = root.val

        while traversal_queue:
            size = len(traversal_queue)
            for i in range(size):
                node = traversal_queue.popleft()
                
                # Record the first node of each level
                if i == 0:
                    left_bottom_value = node.val

                if node.left:
                    traversal_queue.append(node.left)
                if node.right:
                    traversal_queue.append(node.right)
        return left_bottom_value

```

## Path Sum

**Solution**:

1. **Recursive Traversal:**
   - Use a helper function (`traversal`) to recursively traverse the tree, keeping track of the current path sum as 
     it progresses down from the root to the leaves.

2. **Base Case - Leaf Node Check:**
   - When a leaf node (a node with no left or right children) is reached, check if the accumulated path sum plus 
     the leaf node's value equals `targetSum`. If it matches, return `True`; otherwise, return `False`.

3. **Recursive Checks for Left and Right Subtrees:**
   - Recursively call the `traversal` function on the left and right subtrees, updating the current path sum with 
     the value of the current node.
   - Use `hasLeft` and `hasRight` to store the results of the recursive checks. If either subtree returns `True`, the function returns `True`, indicating a path exists.

4. **Combine Results Using Logical `or`:**
   - After checking both left and right subtrees, combine the results using `or`. This ensures that if any valid path 
     is found, the function will return `True`.

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        def traversal(root: Optional[TreeNode], pathSum: int, targetSum: int) -> bool:
            if not root:
                return False

            if not (root.left or root.right):
                return True if pathSum + root.val == targetSum else False

            checkLeft, checkRight = False, False
            if root.left:
                # Recursively check the left subtree with an updated path sum.
                # `pathSum + root.val` creates a new value, leaving the original `pathSum` unchanged.
                # This allows implicit backtracking, as each recursive call operates independently.
                checkLeft = traversal(root.left, pathSum + root.val, targetSum)
            if root.right:
                checkRight = traversal(root.right, pathSum + root.val, targetSum)
            return checkLeft or checkRight
        return traversal(root, 0, targetSum)

```

## Construct Binary Tree from Inorder and Postorder Traversal

p.s., `list[:0]` return `[]`

**Solution**:

1. **Base Case for Recursion:**
   - If the `postorder` list is empty, return `None`. This handles the scenario where there are no more nodes to build, effectively terminating that branch of recursion.

2. **Identify the Root Node:**
   - The last element in the `postorder` list (`postorder[-1]`) is always the **root** of the current subtree. 
   - Create a new `TreeNode` with this value.

3. **Handle Single Node Tree:**
   - If the `postorder` list has only one element, return the root node, as there are no left or right subtrees to process.

4. **Divide Inorder List to Find Left and Right Subtrees:**
   - Locate the `rootValue` in the `inorder` list using `inorder.index(rootValue)`. The index divides the `inorder` list into:
     - **Left Subtree:** Elements before `rootIndex`.
     - **Right Subtree:** Elements after `rootIndex`.

5. **Match Left and Right Subtrees in Postorder List:**
   - Use the length of `left_inorder` to correctly slice the `postorder` list:
     - **Left Subtree:** `postorder[:len(left_inorder)]` corresponds to the left subtree.
     - **Right Subtree:** `postorder[len(left_inorder):-1]` captures the right subtree, excluding the last element (root).

6. **Recursive Construction:**
   - Recursively call `buildTree` to construct the left and right subtrees using the respective `inorder` and `postorder` segments.
   - Attach the resulting left and right subtrees to the `root` node.

7. **Return the Constructed Tree:**
   - Once the recursive calls complete, return the `root`, which now has its left and right subtrees correctly attached.

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not postorder:
            return None

        rootValue = postorder[-1]
        root = TreeNode(rootValue)
        if len(postorder) == 1:
            return root

        rootIndex = inorder.index(rootValue)

        leftInorder, rightInorder = inorder[:rootIndex], inorder[rootIndex+1:]
        leftPostorder, rightPostorder = postorder[:len(leftInorder)], postorder[len(leftInorder):-1]

        root.left = self.buildTree(leftInorder, leftPostorder)
        root.right = self.buildTree(rightInorder, rightPostorder)

        return root

```

## Construct Binary Tree from Preorder and Inorder Traversal

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        def helper(pre_start, pre_end, in_start, in_end) -> Optional[TreeNode]:
            if pre_start > pre_end or in_start > in_end:
                return None

            # Get the root value from preorder
            value = preorder[pre_start]
            # Find the root's index in inorder
            index = inorder_map[value]

            # Create the TreeNode for the root
            node = TreeNode(val=value)

            # Number of nodes in the left subtree
            left_tree_size = index - in_start

            # Recursively build left and right subtrees
            node.left = helper(pre_start+1, pre_start+left_tree_size, in_start, index-1)
            node.right = helper(pre_start+left_tree_size+1, pre_end, index+1, in_end)

            return node

        # Precompute the index map for inorder to achieve O(1) lookups
        inorder_map = {value: index for index, value in enumerate(inorder)}
        
        # Initialize the helper function with full range
        return helper(0, len(preorder)-1, 0, len(inorder)-1)

```

## Maximum Binary Tree

**Solution**:

```python
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None

        rootValue = max(nums)
        root = TreeNode(rootValue)

        if len(nums) == 1:
            return root

        rootIndex = nums.index(rootValue)

        leftNums = nums[:rootIndex]
        rightNums = nums[rootIndex+1:]

        root.left = self.constructMaximumBinaryTree(leftNums)
        root.right = self.constructMaximumBinaryTree(rightNums)
        return root
        
```

## Merge Two Binary Tree

**Key Concepts**

1. **Recursive Approach**:
    - The solution uses recursion to traverse both trees simultaneously. At each recursive step, the function merges the current nodes of `root1` and `root2` by adding their values.
    - The recursion continues for the left and right children of the nodes, effectively traversing both trees in a synchronized manner.

2. **Base Cases**:
    - **`root1` is `None`**: If `root1` is `None`, it returns `root2`. This means that when there is no node in `root1`, the corresponding node in `root2` will be used in the merged tree.
    - **`root2` is `None`**: If `root2` is `None`, it returns `root1`. This means that when there is no node in `root2`, the corresponding node in `root1` will be used in the merged tree.

3. **Merging Nodes**:
    - When both `root1` and `root2` are not `None`, their values are added together (`root1.val += root2.val`), and this value is assigned to the merged node.
    - The function recursively merges the left children (`root1.left` and `root2.left`) and assigns the result to `root1.left`.
    - Similarly, it recursively merges the right children (`root1.right` and `root2.right`) and assigns the result to `root1.right`.

4. **Return the Merged Tree**:
    - The merged tree is constructed by modifying `root1` in place. After the recursive process completes, `root1` contains the merged result and is returned.

**Complexity Analysis**

- **Time Complexity**: `O(n)`, where `n` is the minimum number of nodes between `root1` and `root2`. Each node is visited once during the recursion.
- **Space Complexity**: `O(h)`, where `h` is the height of the smaller tree, due to the recursive call stack.

```python
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]):
        if not root1:
            return root2
        if not root2:
            return root1

        root1.val += root2.val

        root1.left = self.mergeTrees(root1.left, root2.left)
        root2.right = self.mergeTrees(root1.right, root2.right)
        return root1
```

## Search in Binary Search Tree

**Key Concepts**

1. **Binary Search Tree (BST) Property**:
    - In a BST, the left child of a node always has a smaller value than the node, and the right child has a larger value.
    - This property allows efficient searching by skipping half of the nodes at each step, similar to binary search on a sorted array.

2. **Recursive Approach**:
    - The function employs recursion to traverse the tree and locate the target value.
    - At each node, the function checks if the current node's value matches `val`:
        - If **equal**, the node is returned.
        - If **less than `val`**, the search continues in the right subtree.
        - If **greater than `val`**, the search continues in the left subtree.

3. **Base Case**:
    - If the `root` is `None`, it means the search has reached a leaf node without finding the target value, so the function returns `None`.

**Complexity Analysis**

- **Time Complexity**: `O(h)`, where `h` is the height of the BST. In the best case (balanced tree), this is `O(log n)`, and in the worst case (unbalanced tree), this can be `O(n)`.
- **Space Complexity**: `O(h)` due to the recursive call stack. This is `O(log n)` in a balanced tree and `O(n)` in the worst case of an unbalanced tree.

```python
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return None

        if root.val == val:
            return root
        elif root.val < val:
            return self.searchBST(root.right, val)
        else:  # root.val > val
            return self.searchBST(root.left, val)
        
```

## Validate Binary Search Tree

**Solution**:
1. **In-Order Traversal**:
    - In a valid BST, the in-order traversal should produce values in strictly ascending order.
    - By maintaining the previous node (`pre`) visited during the traversal, we can check if the current node’s value is greater than the last visited node's value, which ensures the BST property.

2. **Using a Mutable Object for `pre`**:
    - The function uses a list (`pre = [None]`) to keep track of the previously visited node across recursive calls.
    - Lists in Python are mutable, so changes to `pre[0]` inside the recursive function persist across all recursive levels.
    - This solves the issue of maintaining a consistent reference to the previous node during traversal, ensuring that comparisons are accurate.

Complexity Analysis

- **Time Complexity**: `O(n)`, where `n` is the number of nodes in the tree. Each node is visited exactly once during the in-order traversal.
- **Space Complexity**: `O(h)`, where `h` is the height of the tree, due to the recursive call stack. For a balanced tree, this is `O(log n)`, and for a completely unbalanced tree, it can be `O(n)`.


```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def isValid(root: Optional[TreeNode], pre: List[Optional[TreeNode]]) -> bool:
            if not root:
                return True
            
            isLeftValid = isValid(root.left, pre)

            if pre[0] and pre[0].val >= root.val:
                return False
            pre[0] = root

            isRightValid = isValid(root.right, pre)

            return isLeftValid and isRightValid

        pre = [None]
        return isValid(root, pre)

```

## Minimum Absolute Difference in BST

**Solution**:
1. **In-Order Traversal**:
   - Visits nodes in **sorted order** for a BST.
   - Efficiently finds the minimum difference by comparing **consecutive nodes**.

2. **State Tracking**:
   - Use `pre` (previous node) and `minDiff` (minimum difference) as **mutable lists** to maintain state across recursive calls.

3. **Recursive Logic**:
   - Traverse **left subtree**, process **current node**, then traverse **right subtree**.
   - Update `minDiff` using `min(minDiff[0], abs(root.val - pre[0].val))` if `pre[0]` exists.
   - Set `pre[0] = root` to track the current node for the next comparison.

```python
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        def traversal(root: Optional[TreeNode], pre: List[Optional[TreeNode]], minimumDifference: List[int]):
            if not root:
                return
            
            traversal(root.left, pre, minimumDifference)

            if pre[0] and minimumDifference[0] and abs(root.val - pre[0].val) < minimumDifference[0]:
                minimumDifference[0] = abs(root.val - pre[0].val)
            if pre[0] and not minimumDifference[0]:
                minimumDifference[0] = abs(root.val - pre[0].val)
            pre[0] = root
            
            traversal(root.right, pre, minimumDifference)
        
        pre = [None]
        minimumDifference = [None]
        traversal(root, pre, minimumDifference)
        return minimumDifference[0]

```

## Find Mode in Binary Search Tree

**Solution**:

1. **In-Order Traversal**:
   - **Purpose**: Visits nodes in **sorted order** for a Binary Search Tree (BST).
   - **Benefit**: Consecutive identical values are processed together, making it easy to count duplicates and identify the mode(s).

2. **Tracking State Across Traversal**:
   - **`self.pre`**: Stores the **previously visited node**. Used to compare the current node value and update the count if they match.
   - **`self.count`**: Tracks the **current frequency** of the node value being processed.
   - **`self.maxCount`**: Maintains the **highest frequency** encountered, which is needed to identify the mode(s).
   - **`self.result`**: Holds the list of values that occur **most frequently** (the mode(s)).

3. **Counting Logic**:
   - **If `root.val == self.pre.val`**: Increment `self.count` since the current node matches the previous one.
   - **If `root.val != self.pre.val`**: Reset `self.count` to `1` as a new value is being processed.
   - **If `self.pre` is `None`**: Initialize `self.count` for the first node.

4. **Updating Modes**:
   - **When `self.count == self.maxCount`**: Append the current value to `self.result`, as it matches the highest frequency found.
   - **When `self.count > self.maxCount`**: Clear `self.result`, update `self.maxCount`, and add the current value, as a new mode with a higher frequency is found.

5. **Complexity**:
   - **Time Complexity**: \(O(n)\), where \(n\) is the total number of nodes. Each node is visited once during the traversal.
   - **Space Complexity**: \(O(h)\), where \(h\) is the height of the tree, accounting for the recursion stack depth.

```python
class Solution:
    def __init__(self):
        self.pre = None
        self.count = 0
        self.maxCount = 0
        self.result = []

    def findMode(root: Optional[TreeNode]):
        def traversal(cur: Optional[TreeNode]):
            if not cur:
                return

            # In-order traversal: left subtree
            traversal(cur.left)

            # Current node processing logic
            if self.pre and self.pre.val == cur.val:
                self.count += 1
            else:  # self.pre is None / self.pre.val != cur.val
                self.count = 1
            self.pre = cur

            if self.count == self.maxCount:
                self.result.append(cur.val)
            elif self.count > self.maxCount:
                self.result.clear()
                self.maxCount = self.count
                self.result.append(cur.val)

            # In-order traversal: right subtree
            traversal(cur.right)
        
        traversal(root)
        return self.result
        
```

## Lowest Common Ancestor

**Solution**:

1. **Recursive Approach**:
    - The solution uses recursion to traverse the binary tree. At each node, it checks if it can find the nodes `p` and `q` in the left and right subtrees.
    - The goal is to determine the lowest node in the tree from which both `p` and `q` can be reached, making it the Lowest Common Ancestor (LCA).

2. **Base Case**:
    - If the current `root` is `None`, it returns `None` because there is no tree to search.
    - If the current `root` matches either `p` or `q`, it returns `root` because one of the nodes (`p` or `q`) has been found.

3. **Recursive Exploration**:
    - The function recursively searches the left and right subtrees for `p` and `q`:
        - `leftChild = self.lowestCommonAncestor(root.left, p, q)`
        - `rightChild = self.lowestCommonAncestor(root.right, p, q)`

4. **Determine the LCA**:
    - **Case 1**: If `leftChild` and `rightChild` are both non-null, it means `p` and `q` were found in different subtrees. Therefore, the current `root` is the Lowest Common Ancestor.
    - **Case 2**: If only `leftChild` is non-null, it means both `p` and `q` are in the left subtree. So, return `leftChild`.
    - **Case 3**: If only `rightChild` is non-null, it means both `p` and `q` are in the right subtree. So, return `rightChild`.
    - **Case 4**: If both `leftChild` and `rightChild` are `None`, return `None` because neither `p` nor `q` was found.

5. **Time Complexity**:
    - The time complexity is **O(N)**, where `N` is the number of nodes in the tree. In the worst case, the algorithm might visit all nodes.

6. **Space Complexity**:
    - The space complexity is **O(H)**, where `H` is the height of the tree. This is due to the recursion stack. In the worst case (a skewed tree), it can be as large as `O(N)`, but for a balanced tree, it would be `O(log N)`.

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q:
            return root

        leftChild = self.lowestCommonAncestor(root.left, p, q)
        rightChild = self.lowestCommonAncestor(root.right, p, q)

        # If both leftChild and rightChild are found, root is the LCA
        if leftChild and rightChild:
            return root

        # Otherwise, return the non-null child (either leftChild or rightChild)
        # if both are None, return None
        return leftChild if leftChild else rightChild

```

p.s., 

**Forward References in Python Type Annotations**

- **Definition**: A forward reference allows you to use a type in a function or method before the type is actually defined in the code. 
- **Syntax**: Enclose the type name in quotes (e.g., `'TreeNode'`), signaling to Python that it should resolve the type later.
- **When to Use**:
  - If the class or type you're referring to is defined **later** in the code.
  - Useful in situations involving **circular dependencies** or when the type isn't available at the time of parsing.
- **Alternative (Python 3.7+)**: 
  - Use `from __future__ import annotations` at the top of the file to **automatically** treat all type hints as forward references, removing the need for quotes.


## Insert Into A Binary Search Tree

**Solution**:
```python
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        # Base case: if root is None, create a new TreeNode with the value
        if not root:
            return TreeNode(val)
        
        # Recursive insertion based on BST property
        if val < root.val:
            root.left = self.insertIntoBST(root.left, val)
        else:
            root.right = self.insertIntoBST(root.right, val)

        # Return the root node
        return root

```

## Delete Node in BST

1. **Recursive Approach**:
    - The solution uses recursion to locate and delete the specified node (`key`) in the Binary Search Tree (BST).
    - The search follows the BST property: if the `key` is less than the current node's value, move to the left subtree; if greater, move to the right subtree.

2. **Three Cases When Deleting a Node**:
    - **Case 1**: The node to delete has **no children** (a leaf node).
        - Simply remove the node by returning `None`.
    - **Case 2**: The node to delete has **one child**.
        - Replace the node with its child by returning `root.left` or `root.right`, whichever is not `None`.
    - **Case 3**: The node to delete has **two children**.
        - Replace the node's value with the **in-order successor** (smallest value in the right subtree).
        - Delete the in-order successor node from the right subtree to maintain the BST structure.

3. **Finding the In-Order Successor**:
    - Use a helper function (`getMin`) to locate the smallest node in the right subtree.
    - This node is guaranteed to be greater than all nodes in the left subtree and less than or equal to all nodes in the right subtree.

```python
class Solution:
    def getMin(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        cur = root.right
        while cur.left:
            cur = cur.left
        return cur

    def deleteNode(self, root: Optional[TreeNode], key: int):
        if not root:
            return root

        # If the key is less than the root's value, search in the left subtree
        if key < root.val:
            root.left = self.deleteNode(root.left, key)
        # If the key is greater than the root's value, search in the right subtree
        elif key > root.val:
            root.right = self.deleteNode(root.right, key)
        else:
            # Node to be deleted found
            if not (root.left or root.right):
                return None
            if not root.left and root.right:
                return root.right
            if root.left and not root.right:
                return root.left
            
           # Node with two children: find the in-order successor (smallest in the right subtree)
            minNode = self.getMin(root.right)
            # Replace the root's value with the in-order successor's value
            root.val = minNode.val
            # Delete the in-order successor
            root.right = self.deleteNode(root.right, minNode.val)
        return root

```

## Trim A Binary Search Tree

**Solution**:
1. **Purpose**:
    - The goal of the `trimBST` function is to modify a Binary Search Tree (BST) so that all its nodes fall within a specified range `[low, high]`. Any nodes outside this range are removed from the tree.

2. **Recursive Approach**:
    - The function uses recursion to traverse the tree, trimming nodes that fall outside the specified range.
    - It follows the Binary Search Tree property, allowing it to efficiently determine which parts of the tree to keep or remove.

3. **Three Main Scenarios**:
    - **Case 1: Node Value Less Than `low`**:
        - If `root.val < low`, it means all nodes in the left subtree will also be less than `low` (due to BST properties).
        - Therefore, the function should **discard the left subtree** and recursively trim the right subtree.
    - **Case 2: Node Value Greater Than `high`**:
        - If `root.val > high`, it means all nodes in the right subtree will also be greater than `high`.
        - The function should **discard the right subtree** and recursively trim the left subtree.
    - **Case 3: Node Value Within Range**:
        - If `low <= root.val <= high`, the current node is within the range.
        - The function will recursively trim both the left and right subtrees, keeping the current node.

4. **Time and Space Complexity**:
    - **Time Complexity**: \(O(N)\), where `N` is the number of nodes in the tree. In the worst case, every node might need to be visited.
    - **Space Complexity**: \(O(H)\), where `H` is the height of the tree. This is due to the recursion stack. In the worst case (a skewed tree), it can be (O(N)). For a balanced tree, it will be (O(\log N)).

```python
class Solution:
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        if not root:
            return root

        if low <= root.val <= high:
            # If the current node is within the range, recursively trim the left and right subtrees
            root.left = self.trimBST(root.left, low, high)
            root.right = self.trimBST(root.right, low, high)
        if root.val < low:
            # If the current node's value is less than 'low', trim the left subtree and return the right subtree
            return self.trimBST(root.right, low, high)
        if root.val > high:
            # If the current node's value is greater than 'high', trim the right subtree and return the left subtree
            return self.trimBST(root.left, low, high)
        
        return root

```

## Sorted Array To BST

**Solution**:
1. **Recursive Approach**:
    - The solution uses recursion to divide the sorted array into smaller subarrays and create a height-balanced Binary Search Tree (BST).
    - It selects the middle element of the current subarray as the root to ensure the tree remains balanced.

2. **Base Case**:
    - The recursion terminates when `start` is greater than `end`, meaning there are no elements left to process. In this case, `None` is returned, creating leaf nodes.

3. **Balanced Tree Construction**:
    - By choosing the middle element as the root, the function guarantees that the left subtree contains elements smaller than the root, and the right subtree contains elements larger than the root.
    - Recursively applying this process to the left and right subarrays results in a balanced BST.

4. **Time and Space Complexity**:
    - **Time Complexity**: \(O(N)\), where `N` is the number of elements in the array. Each element is processed once.
    - **Space Complexity**: (O(\log N)) on average due to the recursion stack depth, which is proportional to the height of the balanced tree. In the worst case, it can be \(O(N)\) for a skewed tree.

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
       # Helper function to recursively build the BST
        def buildBST(start: int, end: int) -> Optional[TreeNode]:
            if start > end:
                return None
            
            # Choose middle element as the root for balanced BST
            mid = (start + end) // 2
            root = TreeNode(nums[mid])
            
            # Recursively build left and right subtrees
            root.left = buildBST(start, mid - 1)
            root.right = buildBST(mid + 1, end)

            return root
        
        return buildBST(0, len(nums) - 1)

```

## Convert BST to Greater Tree

**Solution**:
1. **Reverse In-Order Traversal**:
    - The solution uses **reverse in-order traversal** (right-root-left) to process nodes in descending order.
    - This traversal order allows the function to accumulate a running sum as it moves from the largest to the smallest node.

2. **Cumulative Sum Tracking**:
    - The `self.sum` attribute keeps a cumulative sum of all node values processed so far.
    - Each node's value is updated to this cumulative sum, so it reflects the sum of all greater or equal values in the BST.

3. **Base Case**:
    - The function checks if `root` is `None`. If it is, it returns `None`, allowing the recursion to terminate.

4. **Time and Space Complexity**:
    - **Time Complexity**: \(O(N)\), where `N` is the number of nodes in the tree, as each node is visited exactly once.
    - **Space Complexity**: \(O(H)\), where `H` is the height of the tree, due to the recursion stack. In the worst case (a skewed tree), it can be \(O(N)\), but for a balanced tree, it will be (O(log N)).

```python
class Solution:
    def __init__(self):
        self.sum = 0

    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None

        root.right = self.convertBST(root.right)

        self.sum += root.val
        root.val = self.sum

        root.left = self.convertBST(root.left)
        
        return root

```  

# Backtracking

1. **Make a Choice**:
   - Start by choosing an element, position, or path in the solution space. This choice represents a possible step toward building a solution.
   - **Example**: In generating combinations, pick an element to include in the current combination.

2. **Explore Further with Recursion**:
   - Recursively explore possibilities after making the initial choice. This step dives deeper into the solution space, building upon the current choice.
   - **Example**: After adding an element to the current combination, recursively add further elements.

3. **Backtrack (Undo the Choice)**:
   - If the current path does not lead to a solution or all possibilities under this choice have been explored, undo the choice (backtrack).
   - This step returns to a previous state, allowing exploration of alternative paths.
   - **Example**: Remove the last element from the combination to try the next possible element in the sequence.

## Combination

**Solution**:
```python
class Solution:
    def combination(self, n: int, k: int) -> List[List[int]]:
        def backtracking(start: int, path: List[int]):
            # Base case: if the combination is of length k, add it to results
            if len(path) == k:
                result.append(path[:])  # Use a copy of path to avoid reference issues
                return

            for i in range(start, n+1):  ## for i in range(start, n-(k-len(path)) + 1 +1)
                # Include i in the current combination
                path.append(i)
            
                # Recur with the next number
                backtrack(i + 1, path)
            
                # Backtrack: remove the last element to explore a new combination
                path.pop()
        result = []
        backtracking(1, [])
        return result

```

`for i in range(start, n-(k-len(path)) + 1 +1)` is used to limit the range in the loop, ensuring efficient backtracking by stopping early when there aren’t enough remaining elements to form a valid combination.

1. **`k - len(path)`**:
   - Represents the **number of additional elements** needed to complete a combination of length `k`.

2. **`n - (k - len(path))`**:
   - Calculates the **last possible starting position** that leaves enough elements to complete a valid combination.
   - Ensures that further iterations don't attempt to form incomplete combinations.

3. **`+ 1 + 1` Adjustment**:
   - Adds `+ 1` to make the end of the range inclusive.
   - Adds another `+ 1` for Python's `range` behavior, giving `+ 2` in total.

## Combination Sum III

1. **Backtracking Approach**:
   - The solution employs a backtracking function, `backtracking`, which incrementally builds each combination.
   - `start` specifies the starting point for the next selection, ensuring that numbers are not reused.

2. **Base Cases**:
   - **Successful Combination**: If the length of `path` is `k` and the sum of `path` is `n`, the combination is valid. We add a copy of `path` to `result`.
   - **Early Exit**: If the sum of `path` exceeds or equals `n` before reaching `k` elements, the function returns early, avoiding unnecessary computation.

3. **Loop Range Optimization**:
   - The loop range `for i in range(start, 10 - (k - len(path)) + 1)` limits iterations by accounting for the remaining numbers needed to reach a length of `k`.
   - This optimization helps prevent redundant recursive calls, improving performance by skipping combinations that cannot reach the desired length.

4. **Backtracking (Undo the Choice)**:
   - After adding a number `i` to `path`, we make a recursive call with `i + 1` to continue building the combination.
   - Once the recursive call completes, `path.pop()` removes the last number added, allowing exploration of alternative paths.

5. **Result Collection**:
   - `result` is a list that accumulates all valid combinations. At the end of the function, `result` contains every unique combination of `k` numbers that add up to `n`.

6. **Time Complexity**:
   - The time complexity is approximately \(O(2^9)\) because each number can either be included or excluded, but optimizations reduce the number of calls.

7. **Space Complexity**:
   - The space complexity is \(O(k)\) for the depth of the recursion stack and \(O(C)\) for storing all valid combinations, where \(C\) is the number of valid solutions.

```Python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        def backtracking(start: int, path: List[int]):
            if len(path) == k and sum(path) == n:
                result.append(path[:])
                return
            if sum(path) >= n:
                return 

            for i in range(start, 10 - (k - len(path)) + 1):
                
                path.append(i)

                backtracking(i+1, path)

                path.pop()
            
        result = []
        backtracking(1, [])
        return result
        
```

## Letter Combinations of a Phone Number 

**Solution**:
1. **Backtracking Approach**:
   - **Backtracking** is used to explore all potential letter combinations by incrementally building each combination and backtracking when a path has been fully explored.
   - The `backtracking` function adds letters one by one to a temporary list (`path`) and moves to the next digit until the length of `path` matches the length of `digits`.

2. **Recursive Base Case**:
   - When the length of `path` is equal to `len(digits)`, a complete combination has been formed.
   - This combination is joined into a string and added to `result`, which holds all valid combinations.

3. **Mapping of Digits to Letters**:
   - A dictionary, `mapping`, defines the relationship between digits and their corresponding letters.
   - For each digit in `digits`, `mapping[digits[index]]` provides the list of letters for that digit, enabling a straightforward way to retrieve possible letters.

4. **Backtracking with Recursive Calls**:
   - For each letter mapped to the current digit, the letter is appended to `path`, and a recursive call is made to continue building the combination with the next digit.
   - After the recursive call returns, `path.pop()` undoes the last addition, preparing `path` for the next possible letter in the current digit’s set, effectively “backtracking.”

5. **Time and Space Complexity**:
   - **Time Complexity**: \(O(4^n)\), where \(n\) is the length of `digits`. Each digit has up to 4 letters (e.g., digit "7" maps to "pqrs"), resulting in \(4^n\) possible combinations.
   - **Space Complexity**: \(O(n)\) for the recursion stack depth and \(O(4^n)\) for storing the resulting combinations.

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        # Mapping of digits to their respective letters
        mapping = {
            "2": ["a", "b", "c"],
            "3": ["d", "e", "f"],
            "4": ["g", "h", "i"],
            "5": ["j", "k", "l"],
            "6": ["m", "n", "o"],
            "7": ["p", "q", "r", "s"],
            "8": ["t", "u", "v"],
            "9": ["w", "x", "y", "z"]
        }

        def backtracking(index: int, path: List[str]):
            # If the current path length equals the input digits' length, add the combination to result
            if len(path) == len(digits):
                result.append("".join(path))
                return

            # Retrieve letters for the current digit
            letters = mapping[digits[index]]
            for letter in letters:
                path.append(letter)
                backtracking(index + 1, path)
                path.pop()  # Backtrack to previous state

        # Initialize result list
        result = []
        backtracking(0, [])
        return result
        
```

## Combination Sum

1. **Backtracking Approach**:
   - **Backtracking** is used to explore each possible combination by incrementally building a path (combination) and backtracking when a path cannot reach the target or has met the target.
   - The `backtracking` function tries adding each candidate starting from the current position, allowing numbers to be reused by passing the same `start` index in each recursive call.

2. **Recursive Base Cases**:
   - **Combination Found**: When `currentSum` matches the `target`, a valid combination is found. A copy of `path` is added to `result`.
   - **Exceeds Target**: If `currentSum` exceeds `target`, the function returns early, pruning paths that cannot lead to a valid solution.

3. **Exploration of Candidates**:
   - For each candidate at index `i`, `backtracking` adds `candidates[i]` to `path`, updates `currentSum`, and recursively calls itself to explore further combinations with the updated `currentSum`.
   - By calling `backtracking(i, path, currentSum + candidates[i])`, the function allows the same candidate to be reused within a single combination.

4. **Backtracking (Undo the Choice)**:
   - After each recursive call, `path.pop()` removes the last candidate added, effectively "backtracking" to explore alternative combinations by moving to the next candidate.

5. **Time and Space Complexity**:
   - **Time Complexity**: This approach has exponential time complexity, approximately \(O(2^t)\), where \(t\) is `target` divided by the smallest candidate, due to the large number of potential combinations.
   - **Space Complexity**: \(O(t)\) for the maximum depth of the recursion stack (related to `target`) and \(O(C)\) for storing `C` valid combinations in `result`.

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtracking(start: int, path: List[int], currentSum: int):
            if currentSum == target:
                result.append(path[:])
                return
            if currentSum > target:
                return

            
            for i in range(start, len(candidates)):
                path.append(candidates[i])

                backtracking(i, path, currentSum+candidates[i])

                path.pop()
        result = []
        backtracking(0, [], 0)
        return result

```

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtracking(start: int, path: List[int], currentSum: int):
            # If the current sum matches the target, we found a valid combination
            if currentSum == target:
                result.append(path[:])
                return
            # Prune paths where the sum exceeds the target
            if currentSum > target:
                return

            for i in range(start, len(candidates)):
                # Early exit for any further candidates that would exceed target
                if currentSum + candidates[i] > target:
                    break  # Since candidates are sorted, no need to check further
                
                path.append(candidates[i])
                # Recursive call with `i` to allow reuse of the current element
                backtracking(i, path, currentSum + candidates[i])
                path.pop()  # Backtrack to try the next candidate
        
        result = []
        candidates.sort()  # Sort candidates to enable pruning
        backtracking(0, [], 0)
        return result
```

## Palindrome Partitioning

**Solution**:

1. **Backtracking Approach**:
   - **Backtracking** is used to explore all possible partitions of `s`. Each recursive call tries to extend the current partition with a substring if it’s a palindrome.
   - The `backtracking` function takes the current starting index and builds partitions incrementally until the end of the string is reached.

2. **Recursive Base Case**:
   - When `start` reaches the length of `s`, it means a complete partition has been formed. The current partition (stored in `current_partition`) is added as a copy to the `result` list.

3. **Checking for Palindromes**:
   - A helper function, `isPalindrome`, checks if a substring `s[left:right+1]` is a palindrome by comparing characters from both ends moving toward the center.
   - This function allows the backtracking process to include only palindromic substrings in each partition, ensuring that all partitions meet the palindrome requirement.

4. **Recursive Exploration and Backtracking**:
   - For each starting index `start`, the backtracking function iterates over possible ending indices `i` to check each substring `s[start:i+1]`.
   - If the substring is a palindrome, it is added to the current partition, and a recursive call is made to continue building the partition.
   - After each recursive call, `current_partition.pop()` removes the last added substring, effectively "backtracking" to explore other potential partitions.

5. **Result Collection**:
   - `result` collects all valid palindrome partitions as lists of strings. Each partition represents a unique way to divide `s` into palindromic substrings.

6. **Time and Space Complexity**:
   - **Time Complexity**: Approximately $(O(N \times 2^N))$, where (N) is the length of `s`, due to the recursive generation of all possible partitions and palindrome checks.
   - **Space Complexity**: (O(N)) for the recursion stack depth and (O(2^N)) for storing all partitions in `result`.

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        # Helper function to check if a substring is a palindrome
        def isPalindrome(left: int, right: int) -> bool:
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True

        # Backtracking function to build palindrome partitions
        def backtracking(start: int, path: List[str]):
            if start == len(s):
                result.append(path[:])
                return

            for i in range(start, len(s)):
                # If substring s[start:i+1] is a palindrome
                if isPalindrome(start, i):
                    path.append(s[start:i+1])
                    backtracking(i+1, path)
                    path.pop()  # Backtrack to try other partitions

        result = []
        backtracking(0, [])
        return result 

```

## Restore IP Addresses

1. **Backtracking Approach**:
   - The solution uses **backtracking** to explore all potential ways to divide the string into 4 segments.
   - For each segment, it checks if the segment is valid before proceeding. If a path of 4 valid segments reaches the end of the string, it forms a valid IP address and is added to the result.

2. **Helper Function `isValid`**:
   - **Purpose**: `isValid` checks if a substring represents a valid IP segment.
   - **Conditions**:
     - The segment contains only digits.
     - If the segment has multiple digits, it cannot start with "0".
     - The integer value of the segment must be between 0 and 255.

3. **Recursive Exploration in `backtracking`**:
   - **Base Case**: If `path` has 4 segments and `start` equals the length of `s`, a valid IP address is formed. It’s added to `result`.
   - **Early Termination**: If `path` already has 4 segments but `start` hasn’t reached the end of `s`, the function returns early to prevent invalid paths.
   - **Iterating through Possible Segments**: The loop tries segments of length 1 to 3, since each IP segment can have at most 3 digits.

4. **Optimizations**:
   - **Limit Segment Length**: The loop limits each segment to a maximum of 3 characters, ensuring efficient exploration.
   - **Avoid Redundant Slicing**: `isValid` caches the current substring (`segment = s[left:right+1]`), reducing repeated slicing and improving performance.

5. **Time and Space Complexity**:
   - **Time Complexity**: Approximately \(O(3^4)\), as we try up to 3 characters per segment for a total of 4 segments.
   - **Space Complexity**: \(O(1)\) for the path and segment limits, but additional space is used for storing all valid IP addresses in `result`.

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def isValid(left: int, right: int) -> bool:
            segment = s[left:right+1]
            # Check if segment is a digit-only string
            if not segment.isdigit():
                return False
            # Check for leading zero in multi-digit segment
            if len(segment) > 1 and segment[0] == "0":
                return False
            # Check if segment is within the valid IP range
            if int(segment) > 255:
                return False
            return True

        def backtracking(start: int, path: List[str]):
            # Base case: a valid IP address
            if len(path) == 4 and start == len(s):
                result.append(".".join(path))
                return
            # Early return if path already has more than 4 segments
            if len(path) >= 4:
                return

            # Loop to try segments of length 1 to 3
            for i in range(start, min(start + 3, len(s))):  # limits each segment to a maximum of 3 digits
                if isValid(start, i):
                    path.append(s[start:i+1])  # Add the current segment to path
                    backtracking(i + 1, path)  # Recur for the next part
                    path.pop()  # Backtrack to try the next segment

        result = []
        backtracking(0, [])
        return result

```

## Subsets

**Solution**:

1. **Backtracking Approach**:
   - The **backtracking** function recursively explores all subsets by building paths step by step.
   - At each step, it makes a choice to either include or exclude the current element, thereby exploring all possible combinations.

2. **Recursive Structure**:
   - The `backtracking` function takes in `start`, which is the starting index for the current subset, and `path`, which is the current subset being built.
   - The function adds a copy of `path` to `result` to capture the subset formed so far.

3. **Base Case**:
   - The base case is implicitly handled by the loop and backtracking structure: when `start` exceeds the length of `nums`, the recursion simply terminates.
   - This ensures that all combinations are explored without adding any additional conditions.

4. **Recursive Exploration and Backtracking**:
   - The loop iterates over elements starting from the current index `start` to the end of `nums`.
   - For each element `nums[i]`, it’s added to `path`, and a recursive call is made to continue building the subset from the next index.
   - After the recursive call, `path.pop()` undoes the last addition, effectively "backtracking" to explore the next subset.

5. **Result Collection**:
   - `result` accumulates all subsets, with each subset represented as a separate list in `result`.

6. **Time and Space Complexity**:
   - **Time Complexity**: \(O(2^n)\), where \(n\) is the length of `nums`, as there are \(2^n\) possible subsets.
   - **Space Complexity**: \(O(n)\) for the recursion depth and \(O(2^n)\) for storing all subsets in `result`.

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtracking(start: int, path: List[int]):
            result.append(path[:])

            if start >= len(nums):
                return

            for i in range(start, len(nums)):
                path.append(nums[i])
                backtracking(i+1, path)
                path.pop()


        result = []
        backtracking(0, [])
        return result

```

## Combination Sum II

**Solution**:

1. **Backtracking Approach**:
   - **Backtracking** is used to explore all possible combinations of candidates by incrementally building each combination and pruning invalid paths.
   - Each recursive call considers whether adding a candidate meets or exceeds `target` and only explores valid paths further.

2. **Handling Duplicate Candidates**:
   - The solution sorts `candidates` to handle duplicates easily.
   - By checking `if i > start and candidates[i] == candidates[i - 1]: continue`, the function skips duplicate numbers in the same recursive level to avoid redundant combinations.

3. **Early Pruning with Sorted Candidates**:
   - Since `candidates` is sorted, `if currentSum + candidates[i] > target: continue` skips elements that would exceed `target`, reducing the search space and improving efficiency.
   - Sorting also allows early termination of loops when sums become invalid, avoiding unnecessary recursion.

4. **Recursive Base Cases**:
   - **Valid Combination**: If `currentSum` equals `target`, the current `path` is a valid combination and is added to `result`.
   - **Exceeds Target**: If `currentSum` exceeds `target`, the function returns early, avoiding further exploration of invalid paths.

5. **Edge Cases**:
   - **Empty or Insufficient Candidates**: If `candidates` is empty or if the smallest candidate exceeds `target`, the function returns `[]` immediately.

6. **Time and Space Complexity**:
   - **Time Complexity**: Approximately \(O(2^n)\), where \(n\) is the number of elements in `candidates`, since it explores all subsets with pruning.
   - **Space Complexity**: \(O(target)\) for recursion depth and \(O(2^n)\) to store all valid combinations in `result`.

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtracking(start: int, path: List[int], currentSum: int):
            if currentSum == target:
                result.append(path.copy())
                return
            if currentSum > target:
                return
            
            for i in range(start, len(candidates)):
                # If adding candidates[i] would exceed the target, exit the loop
                if currentSum + candidates[i] > target:
                    break
                # Skip duplicate elements to avoid duplicate combinations
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                
                path.append(candidates[i])
                backtracking(i + 1, path, currentSum + candidates[i])
                path.pop()  # Backtrack to try the next candidate
        
        if not candidates or min(candidates) > target:
            return []
        
        result = []
        candidates.sort()
        backtracking(0, [], 0)
        return result
        
```

## Non-decreasing Subsequences

**Solution**:

1. **Backtracking Approach**

   - The solution uses **backtracking** to explore all possible subsequences of the input list `nums`.
   - Backtracking is well-suited here as it allows the algorithm to build each potential subsequence, explore it, and then backtrack to explore new possibilities.
   - For each recursive call, the function decides whether to include or exclude each element in the current subsequence.

2. **Maintaining Non-Decreasing Order**

   - The function only builds **non-decreasing subsequences**.
   - When iterating over elements in `nums`, it checks if the current element `nums[i]` is greater than or equal to the last element of the current sequence (`path`). If it’s not, it skips that element.
   - This ensures that only valid, non-decreasing subsequences are built.

3. **Using a `used` Dictionary for Duplicate Prevention**

   - At each recursive level, a `used` dictionary tracks elements that have already been added at that level.
   - If an element has already been used at a particular recursion depth, it is skipped to prevent duplicate subsequences.
   - `used` is reset for each recursive call so that duplicate checking is isolated to each level of recursion, preventing the same value from being considered twice within the same subsequence.

4. **Collecting Valid Subsequences**

   - A subsequence is added to the `result` list only if it has a length of 2 or more, ensuring that all subsequences meet the problem's requirements.
   - Each time a valid subsequence is found, a copy of it is appended to `result`.
   - The algorithm uses `.copy()` to prevent later modifications from affecting the stored subsequence.

5. **Recursive Exploration and Backtracking**

   - The algorithm explores each element starting from the current `start` index up to the end of `nums`.
   - It adds each valid element to `path`, makes a recursive call to explore further extensions of the subsequence, and then **backtracks** by removing the last element from `path`.
   - This backtracking step allows the function to explore new subsequences while maintaining previously found subsequences intact.
   - 

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        def backtracing(start: int, path: List[int]):
            if len(path) >= 2:
                result.append(path.copy())
                
            if start >= len(nums):
                return

            used = {}  # Reset `used` at each level
            for i in range(start, len(nums)):
                if (len(path) > 0 and nums[i] < path[-1]) or nums[i] in used:   ## or !!!
                    continue
                
                path.append(nums[i])
                used[nums[i]] = 1  # Mark `nums[i]` as used in this level
                
                backtracing(i + 1, path)
                
                path.pop()  # Backtrack

        result = []
        backtracing(0, [])
        return result
```

## Permutations

**Solution**:


1. Backtracking with Path and Used List
   - **Backtracking** is employed to explore all possible permutations by constructing a path step-by-step.
   - **Path**: A list that keeps track of the current sequence of numbers.
   - **Used List**: A boolean list where `used[i]` indicates whether `nums[i]` is currently in the path.
  
2. Base Case for Complete Permutation
   - When `path` length equals `nums` length, we have a complete permutation.
   - We then add a copy of the `path` to the result list to capture this unique permutation.

3. Skipping Used Elements
  - **Condition**: `if used[i]: continue`
  - If `used[i]` is `True`, it means `nums[i]` is already in the current path, so we skip it to avoid repetition in a single permutation.


4. Recursive Calls and Backtracking
   - **Adding**: We add `nums[i]` to `path` and mark `used[i]` as `True`, then recursively call the function to continue building the permutation.
   - **Backtracking**: After the recursive call, we backtrack by removing the last element from `path` and resetting `used[i]` to `False`, allowing that element to be reused in future branches.

5. Collecting Results
   - When a complete permutation is found, it is added to the `result` list.
   - The final `result` contains all unique permutations of `nums`, pruned of any duplicate paths.

6. Time Complexity
   - The time complexity of this approach is **O(N * N!)**, where `N` is the length of `nums`.
   - The pruning of duplicate paths helps reduce unnecessary recursive calls, making the algorithm more efficient when duplicates are present.

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(path: List[int], used: List[bool]):
            # Base case: if path length is the same as nums, we have a complete permutation
            if len(path) == len(nums):
                result.append(path[:])
                return
            
            for i in range(len(nums)):
                # Skip used elements to avoid reusing the same element in the same permutation
                if used[i]:
                    continue
                
                # Add current element to path and mark as used
                path.append(nums[i])
                used[i] = True
                
                # Recurse with updated path and used list
                backtrack(path, used)
                
                # Backtrack: remove current element from path and mark as unused
                path.pop()
                used[i] = False

        result = []
        backtrack([], [False] * len(nums))
        return result
        
```

## Permutations II

**Solution**:

1. Sorting the Input Array
   - **Purpose**: Sorting `nums` helps in identifying and handling duplicate elements.
   - By placing duplicates next to each other, we can efficiently skip redundant branches in the backtracking tree.

2. Backtracking with Path and Used List
   - **Backtracking** is used to explore all possible permutations by incrementally building each path.
   - **Path**: A list representing the current sequence of numbers in the permutation.
   - **Used List**: A boolean list where `used[i]` indicates whether `nums[i]` is currently in the path, preventing reuse of the same element in the current sequence.

3. Base Case for a Complete Permutation
   - When `path` length matches `nums` length, a complete permutation is formed.
   - This permutation (copy of `path`) is added to `result`, capturing the unique permutation.

4. Skipping Used Elements
   - **Condition**: `if used[i]: continue`
   - If `used[i]` is `True`, `nums[i]` is already in the current path, so we skip to avoid repetition.

5. Pruning Duplicate Paths
   - **Condition**: `if i > 0 and nums[i] == nums[i-1] and not used[i-1]: continue`
   - **Explanation**:
       - `nums[i] == nums[i-1]` checks if the current number is a duplicate of the previous one.
       - `not used[i-1]` ensures we skip `nums[i]` if the previous duplicate (`nums[i-1]`) has not been used in the current path (in the same deep level).
   - **Purpose**: This prevents generating duplicate permutations by ensuring only the first occurrence of each duplicate in each level of recursion is used in the current branch.

6. Recursive Calls and Backtracking Steps
   - **Adding**: Add `nums[i]` to `path` and set `used[i]` to `True`, then recursively call `backtracking` to continue building the permutation.
   - **Backtracking**: After the recursive call, we backtrack by removing the last element from `path` and resetting `used[i]` to `False`, allowing that element to be reused in future paths.

7. Collecting Results
   - When a complete permutation is formed, it is appended to `result`.
   - The final `result` contains all unique permutations of `nums`, with duplicate paths pruned out.

8. Time Complexity
   - The time complexity of this approach is **O(N * N!)**, where `N` is the length of `nums`.
   - Pruning duplicate paths reduces unnecessary recursive calls, improving efficiency when duplicates are present.

## Reconstruct Itinerary

**Solution**:

1. Building the Graph (Flight Map)
   - **Data Structure**: A `defaultdict` of lists (`path`) is used to represent the graph, where each key is an airport and the value is a list of destination airports.
   - **Graph Creation**:
        - For each `ticket` in `tickets`, add the destination (`ticket[1]`) to the list of the departure airport (`ticket[0]`).
        - **Sorting**: Each list of destinations is sorted in **reverse lexicographical order** so that we can use `pop()` to efficiently get the smallest lexicographical destination during traversal.

2. Depth-First Search (DFS) Traversal
   - **Purpose**: To explore all paths and build the itinerary while using each ticket exactly once.
   - **Recursive Function**:
       - **`dfs(node)`**:
       - While there are remaining destinations from the current `node` in `path[node]`, recursively call `dfs` on the last destination using `pop()`.
       - Append the `node` to `result` after all destinations from that `node` are exhausted (post-order traversal).
       - **Efficiency**: Using `pop()` ensures each operation is **O(1)**, and sorting in reverse order helps maintain the lexicographical order when traversing.

4. Constructing the Result Itinerary
   - **Appending in Reverse**: Nodes are appended to `result` only after all their outgoing flights are used, resulting in the itinerary being built in reverse order.
   - **Reversing `result`**: The final itinerary is obtained by reversing `result` (`result[::-1]`), providing the correct order from start to finish.

```python
from typing import List
from collections import defaultdict

class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        def dfs(node: str):
            while path[node]:
                dfs(path[node].pop())
            result.append(node)

        path = defaultdict(list)
        # Step 1: Build the graph using a defaultdict of lists
        for ticket in tickets:
            path[ticket[0]].append(ticket[i])
        # Step 2: Sort each list of destinations in reverse lexicographical order
        # This allows us to use pop() to get the next smallest lexicographical destination
        for node in path.keys():
            path[node].sort(reverse=True)

        result = []
        dfs('JFK')
        return result[::-1]

```

## N-Queens

**Solution**:

1. Approach: Backtracking
   - The solution uses a backtracking algorithm to explore possible placements of queens row by row.
   - The `backtracing` function recursively attempts to place queens on the board and backtracks when a placement leads to an invalid configuration.

2. Validity Check (`isValid` Function)
   - Ensures that placing a queen at `chessboard[row][col]` does not conflict with:
     - **Vertical Attack**: Checks all rows above the current one to ensure no queen is in the same column.
     - **Left Diagonal Attack**: Checks the upper left diagonal for any existing queen.
     - **Right Diagonal Attack**: Checks the upper right diagonal for any existing queen.
   - The function iterates over relevant positions to validate if placing a queen is safe.

3. Chessboard Representation
   - The board is represented as a 2D list initialized with `'.'` to denote empty spaces.
   - Queens are placed using `'Q'`, and positions are reset to `'.'` when backtracking.

4. Result Construction
   - When a valid configuration is found (i.e., a queen is placed in every row), the board is converted into a list of strings and appended to `result`.
   - The final `result` contains all possible valid configurations of placing `n` queens on the board.

5. Backtracking Logic
   - The `backtracing` function iterates over columns for each row.
   - If placing a queen at a column is valid (`isValid` returns `True`):
     - The queen is placed (`chessboard[row][i] = 'Q'`).
     - The function recursively calls itself for the next row.
     - The queen is removed (`chessboard[row][i] = '.'`) after the recursive call to explore other placements.

6. Time Complexity
   - The worst-case time complexity is **O(N!)** due to the exponential number of possible placements.
   - The backtracking approach optimizes by pruning invalid paths early.

7. Space Complexity
   - The space complexity is **O(N^2)** for the chessboard and **O(N)** for the recursion stack.

Code Structure
- **`isValid` Function**: Checks if placing a queen at the current position is valid.
- **`backtracing` Function**: Recursively places queens and backtracks when necessary.
- **Result**: The final list of solutions, where each solution is a valid configuration of queens on the board.

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def isValid(row: int, col: int, chessboard: List[List[str]]) -> bool:
            for i in range(row):
                if chessboard[i][col] == 'Q':
                    return False
            for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
                if chessboard[i][j] == 'Q':
                    return False
            for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
                if chessboard[i][j] == 'Q':
                    return False
            return True

        def backtracing(row: int, chessboard: List[List[str]]):
            if row == n:
                result.append([''.join(r) for r in chessboard])
                return

            for i in range(n):
                if isValid(row, i, chessboard):
                    chessboard[row][i] = 'Q'
                    backtracing(row+1, chessboard)
                    chessboard[row][i] = '.'

        result = []
        chessboard = [['.' for _ in range(n)] for _ in range(n)]
        backtracing(0, chessboard)
        return result

```

## Sudoku Solver

**Solution**:

1. Problem Overview
   - The Sudoku problem involves filling a 9x9 board so that each row, column, and 3x3 subgrid contains the digits 1 to 9 without repetition.
   - The given board has some pre-filled numbers and empty cells denoted by `'.'`.

2. Approach: Backtracking Algorithm
   - The solution uses a **backtracking algorithm** to explore potential number placements recursively.
   - The algorithm attempts to place each number from 1 to 9 in an empty cell, validates the placement, and proceeds to the next cell.
   - If a placement leads to an invalid state, the algorithm **backtracks** by removing the number and trying the next possibility.

3. `isValid` Function
   - **Purpose**: Checks if placing a given number at `board[rowNum][colNum]` is valid according to Sudoku rules.
   - **Row and Column Check**:
      - Iterates through the specified row and column to ensure the number is not already present.
   - **3x3 Subgrid Check**:
       - Calculates the starting indices of the 3x3 subgrid using integer division (`rowNum // 3 * 3` and `colNum // 3 * 3`).
   - Iterates through the subgrid to ensure the number is not present.
   - **Return**:
      - Returns `True` if the number can be placed without breaking any rules; otherwise, `False`.

4. `backtracing` Function
   - **Purpose**: Fills the board recursively by trying valid numbers in each empty cell.
   - **Logic**:
     - Iterates over each cell in the 9x9 grid.
     - If an empty cell (`'.'`) is found, tries placing each number from 1 to 9.
     - Calls `isValid` to check if the number can be placed.
     - If a valid placement is found, places the number and recursively calls `backtracing` for the next cell.
     - If placing a number leads to a solution, the function returns `True`.
     - If no valid number can be placed, resets the cell to `'.'` and returns `False` to backtrack.
   - **Base Case**:
     - If all cells are filled correctly, returns `True` indicating the board is solved.

5. In-Place Modification
   - The function modifies the board directly without returning it, as per the problem requirement to solve the board **in-place**.

6. Time Complexity
   - **Worst Case**: **O(9^(N^2))**, where `N` is the size of the board (9 for a standard Sudoku).
   - The backtracking approach explores all potential placements but prunes invalid paths early.

7. Space Complexity
   - **O(N^2)** due to the recursion stack in the backtracking process, where `N` is 9 (board size).
   - The `isValid` function does not use additional data structures, keeping space usage minimal.

8. Key Points
   - **Recursive Approach**: Solves the problem by exploring, validating, and backtracking if needed.
   - **Validation Function**: Ensures that each number placement adheres to Sudoku rules.
   - **Backtracking**: Essential for handling incorrect placements and exploring alternate solutions.
   - **Efficiency**: Prunes paths as soon as an invalid state is detected, improving performance.

```python
class Solution:
     def solveSudoku(self, board: List[List[str]]) -> None:
        def isValid(rowNum: int, colNum: int, num: str, board: List[List[str]]) -> bool:
        # Check the row and column
            for i in range(9):
                if board[rowNum][i] == num or board[i][colNum] == num:
                    return False

            # Determine the starting indices of the 3x3 subgrid
            rowStart = (rowNum // 3) * 3
            colStart = (colNum // 3) * 3

            # Check the 3x3 subgrid
            for i in range(rowStart, rowStart + 3):
                for j in range(colStart, colStart + 3):
                    if board[i][j] == num:
                        return False

            return True
        
        def backtracing(board: List[List[str]]) -> bool:
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        for num in range(1, 10):
                            if isValid(i, j, str(num), board):
                                board[i][j] = str(num)
                                if backtracing(board):
                                    return True
                                board[i][j] = '.'  # Backtrack
                        return False  # If no valid number, return False
            return True  # All cells are filled correctly

        backtracing(board)  # Start the backtracking process

```

# Greedy Algorithm

## Assign Cookies

**Solution**:

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        # Sort the greed factors of children in ascending order
        g.sort()  
        # Sort the sizes of cookies in ascending order
        s.sort()  
        
        # Start with the largest cookie index
        index = len(s) - 1  
        result = 0  # Initialize the result to count content children
        
        # Iterate over the children starting from the most greedy to the least
        for i in range(len(g) - 1, -1, -1):  
            # Check if there is a cookie available and if it can satisfy the current child's greed
            if index >= 0 and s[index] >= g[i]:  
                result += 1  # Increment the result as the child is content
                index -= 1  # Move to the next largest available cookie
        
        return result  # Return the total number of content children
```

## Wiggle Subsequence

```python
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        # Initialize the length of the wiggle sequence
        result = 1
        prevDiff = 0
        
        for i in range(1, len(nums)):
            diff = nums[i] - nums[i - 1]
            
            # Check if the current difference changes the direction
            if (diff > 0 and prevDiff <= 0) or (diff < 0 and prevDiff >= 0):
                result += 1
                prevDiff = diff  # Update the previous difference
        
        return result

```

## Maximum Subarray

**Solution**:
1. Key Points
   - **Greedy Extension**: Extend the subarray only if it results in a positive sum.
   - **Reset Condition**: Start a new subarray if `preSum` becomes negative, ensuring that negative sums do not reduce the potential `maxSum`.

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # Initialize preSum to track the current subarray sum starting with the first element
        preSum = nums[0]
        # Initialize maxSum to keep track of the maximum subarray sum found so far
        maxSum = nums[0]

        # Iterate through the array starting from the second element
        for i in range(1, len(nums)):
            # If preSum is negative, start a new subarray at the current element
            # Otherwise, add the current element to the existing subarray sum
            preSum = nums[i] if preSum < 0 else preSum + nums[i]
            
            # Update maxSum if the current subarray sum (preSum) is greater than maxSum
            if preSum > maxSum:
                maxSum = preSum
                
        # Return the maximum subarray sum found
        return maxSum

```

## Best Time to Buy and Sell Stock II

**Solution**:

Key Points
- **Greedy Choice**: Only add profit if `prices[i] > prices[i - 1]`. This ensures only profitable trades are considered.
- **Time Complexity**: `O(n)` where `n` is the length of the `prices` list, as it iterates through the list once.
- **Space Complexity**: `O(1)` as no additional space is required apart from simple variables.


```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        return sum(max(prices[i] - prices[i-1], 0) for i in range(1, len(prices)))

```

## Jump Game

**Solution**: 

**Greedy Approach**
   - **Logic:**
     - Iterate through each index and its jump length.
     - If the current index is beyond `max_reach`, return `False` (cannot reach this point).
     - Update `max_reach` to the maximum of its current value and `i + jump`.
     - If `max_reach` reaches or exceeds `last_index`, return `True`.

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        # Corrected the empty list check
        if not nums:
            return True
        
        max_reach = 0 
        last_index = len(nums) - 1
        
        for i, jump in enumerate(nums):
            # If the current index is beyond the maximum reach, return False
            if i > max_reach:
                return False
            
            # Update the maximum reach
            max_reach = max(max_reach, i + jump)
            
            # If we've reached or surpassed the last index, return True
            if max_reach >= last_index:
                return True
        
        # Final check in case the loop finishes without early termination
        return max_reach >= last_index 

```

## Jump Game II

**Solution**:

1. **Greedy Approach**:
   - The solution uses a greedy algorithm to find the minimum number of jumps needed to reach the end of the array by always choosing the farthest reachable index at each step.

2. **Initialization**:
   - `cur_max_reach`: Tracks the current maximum index that can be reached with the current number of jumps.
   - `next_max_reach`: Keeps track of the farthest index that can be reached with an additional jump.
   - `count`: Keeps track of the number of jumps taken.

3. **Iterating Through the Array**:
   - Traverse each index `i` in the array and update `next_max_reach` as the maximum of its current value and `i + nums[i]`, representing how far we can jump from index `i`.
   - Check if `i` equals `cur_max_reach` to determine if the current jump segment has ended:
     - Increment `count` since another jump is required to continue.
     - Update `cur_max_reach` to `next_max_reach` to extend the jump range.
     - Break the loop if `cur_max_reach` reaches or exceeds the last index, as no more jumps are needed.

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        # If the array has one or no elements, no jumps are needed
        if len(nums) <= 1:
            return 0

        cur_max_reach, next_max_reach = 0, 0
        count = 0
        for i in range(len(nums)):
            next_max_reach = max(next_max_reach, i+nums[i])

            if i == cur_max_reach:
                # When we reach the end of the current jump's reach
                count += 1
                cur_max_reach = next_max_reach

                if cur_max_reach >= len(nums)-1:
                    break

        return count

```

## Maximize Sum Of Array After K Negations

```python
class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        heapq.heapify(nums)
        
        while k > 0:
            smallest = heapq.heappop(nums)
            heapq.heappush(nums, -smallest)
            k -= 1

        return sum(nums)
        
```

**Solution**:

1. **Sorting by Absolute Values**:
   - The array `nums` is sorted by the absolute value of its elements in descending order (`nums.sort(key=lambda x: abs(x), reverse=True)`).
   - Sorting by absolute value ensures that elements with the highest magnitude (greatest impact on the sum) are processed first, maximizing the benefit of any changes.

2. **Negating Negative Elements**:
   - The loop iterates through `nums` and negates any negative elements to increase the sum until `k` negations are exhausted.
   - Each negation operation reduces `k` by 1.

3. **Handling Remaining `k`**:
   - After processing all negative elements, if `k` is still greater than 0 and odd, the smallest element (now at the end of the list after sorting by absolute value) is negated again to maximize the sum.
   - This step (`nums[len(nums)-1] *= -1 if k % 2 == 1 else 1`) ensures that the parity of `k` is taken into account:
     - If `k` is even, no further changes are made because negating an element twice results in no net change.
     - If `k` is odd, the last element is flipped to minimize the impact of the final negation.

4. **Return the Sum**:
   - The sum of the modified `nums` is returned as the final result.


```python
class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        # Sort the array by the absolute value of each element in descending order
        nums.sort(key=lambda x: abs(x), reverse=True)

        # Negate negative numbers until k is exhausted or no more negative numbers remain
        for i in range(len(nums)):
            if k > 0 and nums[i] < 0:
                nums[i] = -nums[i]  # Negate the current negative element
                k -= 1  # Decrease k for each negation
        
        # If k is odd, negate the smallest (least impactful) element to maximize the sum
        # This handles the scenario where k is still greater than 0 after processing
        nums[len(nums) - 1] *= -1 if k % 2 == 1 else 1

        # Return the sum of the modified array
        return sum(nums)

```

## Gas Station

**Solution**:

1. **Initialize Variables**:
   - `cur_sum`: Tracks the running gas balance from the current starting station to check if the journey can proceed.
   - `total_sum`: Tracks the total gas balance across all stations to determine if the circuit is possible.
   - `start`: Stores the candidate starting index for a potential full circuit.

2. **Loop Through Each Station**:
   - For each station `i`:
     - Calculate `gas_gain = gas[i] - cost[i]` to get the net gain or loss of gas at that station.
     - Update `cur_sum` with `gas_gain` to keep a running balance from the current `start`.
     - Update `total_sum` with `gas_gain` to track the overall gas balance.

3. **Check for Negative Running Balance (`cur_sum`)**:
   - If `cur_sum` becomes negative, it means we cannot reach the current station `i` from the current `start`.
   - Update `start` to `i + 1` (next station) as the new starting point.
   - Reset `cur_sum` to `0` to start tracking from the new starting station.

4. **Final Check**:
   - If `total_sum` is negative after the loop, it means the total gas is insufficient to complete the circuit, so return `-1`.
   - If `total_sum` is non-negative, return `start` as the starting index from which a complete circuit is possible.

5. **Complexity**:
   - **Time Complexity**: `O(n)` since we only make a single pass through the `gas` and `cost` arrays.
   - **Space Complexity**: `O(1)` as only a few extra variables are used.

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        # Initialize variables to keep track of current sum, total sum, and starting position
        cur_sum, total_sum = 0, 0
        start = 0  # Starting index of the journey

        # Loop through each gas station
        for i in range(len(gas)):
            # Calculate the net gas gain/loss at the current station
            gas_gain = gas[i] - cost[i]
            cur_sum += gas_gain  # Update the running sum for the current segment
            total_sum += gas_gain  # Update the total sum for all stations

            # If the current sum falls below zero, reset the starting position
            # as we can't reach this point from the previous `start`
            if cur_sum < 0:
                start = i + 1  # Set the next station as the new starting point
                cur_sum = 0  # Reset current sum for the new start segment

        # If the total sum is negative, we cannot complete the circuit
        if total_sum < 0:
            return -1

        # If the total sum is non-negative, the journey can be completed
        return start

```

## Candy

**Solution**:

1. **Single Array for Candies**:
   - Use a single array `assigned_candy` initialized with `1` for each child, as each child must have at least one candy.
   
2. **Two-Pass Strategy**:
   - **Left-to-Right Pass**:
     - Traverse the ratings from left to right.
     - For each child, if their rating is higher than the previous child’s rating, increase their candy count by 1 more than the previous child’s candy count.
   - **Right-to-Left Pass**:
     - Traverse the ratings from right to left.
     - For each child, if their rating is higher than the next child’s rating, set their candy count to the maximum of their current count or `assigned_candy[i + 1] + 1`.
     - This ensures that any higher-rated child has more candies than their right neighbor.

3. **Summing Total Candies**:
   - After the two passes, `assigned_candy` contains the correct minimum candies for each child.
   - The total number of candies is the sum of all elements in `assigned_candy`.

4. **Complexity**:
   - **Time Complexity**: `O(n)`, with two linear passes through the array.
   - **Space Complexity**: `O(n)`, for storing candy counts in `assigned_candy`.

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        assigned_candy = [1] * n  # Initialize each child's candy count to 1

        # Left-to-right pass: ensure each child has more candies than the one before if the rating is higher
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                assigned_candy[i] = assigned_candy[i - 1] + 1

        # Right-to-left pass: ensure each child has more candies than the one after if the rating is higher
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                assigned_candy[i] = max(assigned_candy[i], assigned_candy[i + 1] + 1)

        # Sum up the total candies required
        return sum(assigned_candy)
    
```

## Lemonade Change

```python
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five_count, ten_count = 0, 0  # Counters for $5 and $10 bills

        for bill in bills:
            if bill == 5:
                # Accept $5 bill, no change needed
                five_count += 1

            elif bill == 10:
                # Accept $10 bill, need to give back one $5 bill as change
                if five_count >= 1:
                    five_count -= 1
                    ten_count += 1
                else:
                    return False  # Not enough $5 bills to give change

            elif bill == 20:
                # Accept $20 bill, prefer to give back one $10 and one $5 if possible
                if ten_count >= 1 and five_count >= 1:
                    ten_count -= 1
                    five_count -= 1
                elif five_count >= 3:
                    # Otherwise, give three $5 bills as change
                    five_count -= 3
                else:
                    return False  # Not enough bills to give change

        # If we never ran out of change, return True
        return True
    
```

## Queue Reconstruction by Height

**Solution**:

- Sort by height descending and `k` ascending.
- Insert each person in their `k` position in `new_queue`, ensuring all `k` constraints are met.
- Return `new_queue` as the final reconstructed queue.

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        # Edge case: if there's only one person, return the list as is
        if len(people) == 1:
            return people
        
        # Step 1: Sort the people by height in descending order.
        # If two people have the same height, sort them by their 'k' value in ascending order.
        people.sort(key=lambda x: (-x[0], x[1]))
        
        # Initialize an empty list to build the queue
        new_queue = []
        
        # Step 2: Insert each person into the new_queue at the index specified by their 'k' value.
        # Since taller people are placed first, each insertion respects the 'k' value constraint.
        for person in people:
            pos = person[1]  # 'k' is the position index for this person
            new_queue.insert(pos, person)  # Insert person at index 'pos'
        
        return new_queue  # Return the reconstructed queue

```

## Minimum Number of Arrows to Burst Balloons

**Solution**:

- Sort balloons by starting position.
- Track overlapping intervals and minimize the number of arrows by merging overlapping intervals where possible.

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        # Sort points by the starting position of each balloon
        points.sort(key=lambda x: x[0])

        # Start with one arrow as we need at least one to burst the first balloon
        count = 1
        
        # Iterate through each balloon interval starting from the second balloon
        for i in range(1, len(points)):
            # If the current balloon starts after the previous balloon ends, we need a new arrow
            if points[i][0] > points[i-1][1]:
                count += 1  # Increment arrow count
            else:
                # Overlapping balloons can be burst with the same arrow
                # Update the end of the current interval to the minimum of overlapping ends
                points[i][1] = min(points[i-1][1], points[i][1])

        return count  # Return the minimum number of arrows required

```

## Non-overlapping Intervals

**Solution**:

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # Edge case: If there's only one interval, no removals are needed
        if len(intervals) == 1:
            return 0

        # Step 1: Sort intervals by the starting position
        intervals.sort(key=lambda x: x[0])

        # Initialize count of intervals to remove to avoid overlap
        count = 0

        # Step 2: Iterate through each interval starting from the second one
        for i in range(1, len(intervals)):
            # Check if there's an overlap between the current and previous interval
            if intervals[i-1][1] > intervals[i][0]:
                # Increment count as we need to remove one of the overlapping intervals
                count += 1
                # Adjust the end of the current interval to minimize further overlaps
                intervals[i][1] = min(intervals[i-1][1], intervals[i][1])

        return count  # Return the minimum number of intervals to remove

```

## Partition Labels

**Solution**:

1. **Tracking Last Occurrences**:
   - Use a dictionary `last_occurrence` to store the last index of each character in `s`.
   - This dictionary helps quickly find the end boundary for each partition.

2. **Single Pass for Partitioning**:
   - Iterate through `s` while tracking a `right` boundary, which represents the furthest point of the current partition.
   - For each character, update `right` to the maximum of its last known position (`last_occurrence[char]`).
   - When the current index `i` equals `right`, a complete partition is identified.

3. **Storing Partition Lengths**:
   - When a partition boundary is reached (`i == right`), calculate the partition length (`right - left + 1`) and add it to the result list `result`.
   - Move `left` to `i + 1` to start a new partition from the next character.

4. **Return Result**:
   - After iterating through `s`, `result` contains the lengths of each partition, satisfying the required constraints.

5. **Complexity Analysis**:
   - **Time Complexity**: `O(n)`, where `n` is the length of `s`, as we make one pass to build `last_occurrence` and another to partition `s`.
   - **Space Complexity**: `O(1)` auxiliary space, since `last_occurrence` has at most 26 entries for lowercase letters.

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        # Dictionary to store the last occurrence index of each character
        last_occurrence = {char: idx for idx, char in enumerate(s)}

        result = []
        left, right = 0, 0

        # Single pass to determine partitions
        for i, char in enumerate(s):
            # Update `right` to the furthest last occurrence of the current character
            right = max(right, last_occurrence[char])

            # If the current index reaches the `right` boundary, we have a complete partition
            if i == right:
                # Add the partition length to `result`
                result.append(right - left + 1)

                # Move `left` to the next index after the current partition
                left = i + 1

        return result
```

## Merge Intervals

**Solution**:

1. **Sorting by Start Times**:
   - Sort `intervals` by their starting position using `intervals.sort(key=lambda x: x[0])`.
   - Sorting helps position intervals sequentially, allowing us to easily detect and merge overlaps.

2. **Iterative Merging Process**:
   - Initialize `result` with the first interval as a starting reference for merging.
   - For each subsequent interval:
     - **If Overlapping**: If the current interval starts before or when the last merged interval ends (`intervals[i][0] <= result[-1][1]`), merge by updating the end of the last interval in `result` to the maximum end time.
     - **If Not Overlapping**: If the current interval does not overlap, add it as a new interval in `result`.

3. **Return Merged Intervals**:
   - After processing all intervals, `result` contains all merged intervals.

4. **Complexity Analysis**:
   - **Time Complexity**: `O(n log n)` due to sorting.
   - **Space Complexity**: `O(n)`, as we store the merged intervals in `result`.

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])

        result = [intervals[0]]
        for i in range(1, len(intervals)):
            if intervals[i][0] <= result[-1][1]:
                result[-1][1] = max(result[-1][1], intervals[i][1])
            else:
                result.append(intervals[i])
        return result

```

## Monotone Increasing Digits

**Solution**:

1. **Convert the Number to a List of Digits**:
   - Convert the integer `n` to a list of string digits to allow in-place manipulation.

2. **Identify the "Change Point"**:
   - Traverse the list of digits from right to left.
   - For each pair of adjacent digits `(digits[i-1], digits[i])`:
     - If `digits[i-1] > digits[i]`, it indicates a break in the monotonic increase.
     - Decrement `digits[i-1]` by 1 and set `change_point` to `i`.
     - This ensures that all digits after `change_point` will be changed to maintain the monotone property.

3. **Set Trailing Digits to '9'**:
   - For all positions after `change_point`, set digits to `'9'`.
   - This guarantees the largest possible monotone increasing number up to the original value of `n`.

4. **Return the Result as an Integer**:
   - Join the modified list of digits back into a string, convert it to an integer, and return the result.

5. **Complexity**:
- **Time Complexity**: \(O(d)\), where \(d\) is the number of digits in `n`.
- **Space Complexity**: \(O(d)\), for storing the list of digits.

```python
class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        # Convert the number to a list of characters (digits) for easy manipulation
        digits = list(str(n))

        # Initialize change_point to mark where we start setting digits to '9'
        change_point = len(digits)

        # Traverse the number from the end to the beginning
        for i in range(len(digits) - 1, 0, -1):
            # If the current digit is less than the previous one, we need to adjust
            if digits[i - 1] > digits[i]:
                # Decrement the previous digit by 1
                digits[i - 1] = str(int(digits[i - 1]) - 1)
                # Update the change_point to the current index
                change_point = i

        # Set all digits after change_point to '9' to ensure monotonic increase
        for i in range(change_point, len(digits)):
            digits[i] = '9'

        # Convert the list of characters back to an integer and return it
        return int(''.join(digits))
    
```

## Binary Tree Cameras

**Solution**:

1. **Define State Constants**:
   - `COVERED = 0`: Node is covered by a camera but does not have a camera itself.
   - `HAS_CAMERA = 1`: Node has a camera.
   - `NOT_COVERED = 2`: Node is not covered by any camera.

2. **Recursive Traversal Function**:
   - Use a helper function `traversal(cur: TreeNode)` to determine the state of each node.
   - The function recursively evaluates each node’s left and right children, determining the minimal camera placement to cover all nodes.

3. **State Conditions in Traversal**:
   - **If both children are `COVERED`**:
     - Return `NOT_COVERED` for the current node, as it has no camera coverage.
   - **If either child is `NOT_COVERED`**:
     - Place a camera at the current node (`HAS_CAMERA`) and increment `result`.
   - **If either child has a camera (`HAS_CAMERA`)**:
     - Mark the current node as `COVERED`.

4. **Final Camera Check at Root**:
   - After calling `traversal(root)`, if the root is still `NOT_COVERED`, increment `result` to cover the root node with a final camera.

5. **Return the Result**:
   - `result` now holds the minimum number of cameras needed to cover all nodes in the tree.

## Complexity
- **Time Complexity**: `O(n)` where `n` is the number of nodes, as each node is visited once.
- **Space Complexity**: `O(h)` where `h` is the height of the tree, due to recursive call stack depth.

```python
class Solution:
    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        # Define state constants to represent each node's coverage status
        COVERED = 0         # Node is covered but does not have a camera
        HAS_CAMERA = 1      # Node has a camera
        NOT_COVERED = 2     # Node is not covered by any camera

        result = 0  # Initialize camera count to zero

        # Define a helper function to perform post-order traversal on the tree
        def traversal(cur: Optional[TreeNode]) -> int:
            nonlocal result
            # If node is None (base case for leaf children), it is covered by default
            if not cur:
                return COVERED

            # Recursively check the left and right children states
            left_state = traversal(cur.left)
            right_state = traversal(cur.right)

            # Case 1: If both children are COVERED, the current node is NOT_COVERED
            # It will rely on its parent to cover it.
            if left_state == COVERED and right_state == COVERED:
                return NOT_COVERED

            # Case 2: If either child is NOT_COVERED, place a camera at the current node
            # This covers both the node and its children.
            if left_state == NOT_COVERED or right_state == NOT_COVERED:
                result += 1
                return HAS_CAMERA

            # Case 3: If either child has a camera, the current node is covered
            # No need for a camera here.
            if left_state == HAS_CAMERA or right_state == HAS_CAMERA:
                return COVERED

            # This line should not be reached with correct input and logic
            return -1

        # After traversal, check if the root node is covered
        # If root is still NOT_COVERED, add one final camera at the root
        if traversal(root) == NOT_COVERED:
            result += 1

        # Return the total number of cameras needed
        return result
```

p.s.,

| Feature         | `global`                                       | `nonlocal`                                           |
|-----------------|------------------------------------------------|------------------------------------------------------|
| **Scope**       | Refers to the global (module-level) scope      | Refers to the nearest enclosing non-global scope     |
| **Usage**       | Modify/access a module-level variable          | Modify/access a variable in an enclosing function    |
| **Typical Use** | Used when multiple functions need to share module-level state | Used in nested functions to modify the outer function’s variable |
| **Availability**| Available across the entire module             | Limited to the function it’s enclosed in             |

E.g.,
```python
count = 0  # This is a global variable

def increment():
    global count  # Refers to the global 'count' variable
    count += 1

increment()
print(count)  # Output: 1
```

```python
def outer_function():
    count = 0  # This is in the outer function's scope

    def inner_function():
        nonlocal count  # Refers to the 'count' in outer_function
        count += 1

    inner_function()
    return count

print(outer_function())  # Output: 1
```

# Dynamic Programing

**Five Steps to Solve a Dynamic Programming (DP) Problem**

1. Define the Subproblem and State Representation
   - **Identify** the problem's subproblems by breaking down the problem into smaller components.
   - **Define the State**: Determine a state that captures the essence of a subproblem, making it easier to build toward the solution.
   - **Example**: For finding the longest increasing subsequence, define `dp[i]` as the length of the longest subsequence that ends at index `i`.

2. Formulate the Recurrence Relation
   - **Identify how to relate** the current state to previous states.
   - **Create a Recurrence Relation**: Write a formula that defines how each state can be derived from the previous states.
   - **Example**: In the Fibonacci sequence, the recurrence relation is `F(n) = F(n-1) + F(n-2)`.

3. Identify and Initialize Base Cases
   - **Base Cases** provide the starting points to fill in the DP table or cache values in memoization.
   - **Initialize** the smallest subproblem(s) explicitly to allow subsequent states to be computed.
   - **Example**: For Fibonacci, the base cases are `F(0) = 0` and `F(1) = 1`.

4. Choose and Implement the Approach (Top-Down or Bottom-Up)
   - **Top-Down (Memoization)**: Use recursion with a cache to store results as they are computed, which prevents redundant calculations.
   - **Bottom-Up (Tabulation)**: Use an iterative approach to fill up a table from base cases up to the final solution.
   - **Example**: For Fibonacci, you can either use recursive memoization or a loop to calculate values up to `F(n)`.

5. Optimize and Test
   - **Optimize Space** by reducing the dimensions of the DP array if possible (common in problems like Fibonacci).
   - **Test** with various inputs, including edge cases, to ensure accuracy.
   - **Example**: In the knapsack problem, test with cases like small capacities, large weights, and scenarios with zero or a single item to ensure correctness.

## Climbing Stairs

**Solution**:

## Recurrence Relation

To reach step \( n \), you can arrive from:
- Step \( n - 1 \) (by taking a single step).
- Step \( n - 2 \) (by taking a double step).

Thus, the number of ways to reach step \( n \) is the sum of the ways to reach steps \( n - 1 \) and \( n - 2 \).

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        # Base case: if there's only one step, there's only one way to climb it
        if n == 1:
            return 1
        
        # Initialize a DP array with two elements to store ways for the last two steps
        # dp[0] represents the number of ways to reach two steps before
        # dp[1] represents the number of ways to reach the last step
        dp = [1, 1]
        
        # Loop from step 2 up to n, calculating the number of ways to reach each step
        for _ in range(2, n + 1):
            # Calculate the number of ways to reach the current step
            # It's the sum of the ways to reach the previous step (dp[1])
            # and the step before that (dp[0])
            cur = dp[0] + dp[1]
            
            # Update the DP array for the next iteration
            # Shift the values forward: dp[1] becomes the new dp[0], and cur becomes the new dp[1]
            dp[0] = dp[1]
            dp[1] = cur
        
        # dp[1] now holds the number of ways to reach the nth step
        return dp[1]

```

## Min Cost Climbing Stairs

**Solution**:

1. **Dynamic Programming (DP) Approach**:
   - Define a DP array `dp` where `dp[i]` represents the minimum cost to reach the \( i \)-th step.
   - We use the `cost` array to calculate the minimum cost dynamically at each step, considering both the previous one and two steps.

2. **Base Cases**:
   - `dp[0] = cost[0]`: The cost to reach the first step is the cost of that step.
   - `dp[1] = cost[1]`: The cost to reach the second step is the cost of that step.

3. **Recurrence Relation**:
   - To reach step \( i \), you can come from:
     - Step \( i - 1 \) with an additional cost of `cost[i]`.
     - Step \( i - 2 \) with an additional cost of `cost[i]`.
   - Therefore, the minimum cost to reach step \( i \) is:
     \[
     dp[i] = min(dp[i - 1] + cost[i], dp[i - 2] + cost[i])
     \]

4. **Final Result Calculation**:
   - To reach the top of the staircase, you can either arrive from the last step or the second-to-last step, so:
   
   result = min(dp[-1], dp[-2])

- **Recurrence Relation**: \( dp[i] = \min(dp[i - 1] + cost[i], dp[i - 2] + cost[i]) \)
- **Time Complexity**: \( O(n) \), where \( n \) is the number of steps, as we calculate the minimum cost for each step once.
- **Space Complexity**: \( O(n) \), as we store the minimum cost for each step in the DP array.

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # If there are only two steps, return the minimum of the two, as that's the minimum cost to reach the top
        if len(cost) == 2:
            return min(cost)
        
        # Initialize a DP array where dp[i] represents the minimum cost to reach step i
        dp = [0] * len(cost)
        dp[0], dp[1] = cost[0], cost[1]
        
        # Fill the DP array with the minimum cost to reach each step starting from step 2
        for i in range(2, len(cost)):
            # The cost to reach step i is the minimum of reaching it from i-1 or i-2
            dp[i] = min(dp[i - 1] + cost[i], dp[i - 2] + cost[i])
        
        # The result is the minimum cost of reaching the top, which can be done either from the last step
        # or the second-to-last step
        return min(dp[-1], dp[-2])

```

## Unique Paths

**Solution**:
1. **Define the Problem in Terms of Subproblems**:
   - The goal is to find the number of unique paths from the top-left corner to the bottom-right corner of an `m x n` grid.
   - Define `dp[col]` as the number of unique paths to reach the cell in the current row at column `col`.
   - This breaks down the problem into subproblems where each cell's unique path count can be derived from the cell directly above and the cell to the left.

2. **Identify the Recurrence Relation**:
   - The number of unique paths to reach a cell `(row, col)` is the sum of the paths to the cell directly above it and the cell to the left.
   - This gives the recurrence relation: 
     \[
     dp[col] = dp[col] + dp[col - 1]
     \]
   - Here, `dp[col]` initially holds the paths from the row above, and `dp[col - 1]` provides the paths from the left cell.

3. **Define Base Cases**:
   - For the first row or first column, there is only one way to reach each cell (either moving right or moving down).
   - Initialize the `dp` array with `1`s because all cells in the first row of the grid can only be reached in one way.
   
4. **Determine the Iterative Approach**:
   - Use a single 1D array `dp` of size `n` to store the number of unique paths for each column in the current row.
   - Iterate row by row and, for each cell `(row, col)`, update `dp[col]` in place by adding `dp[col - 1]`.
   - This way, `dp[col]` accumulates the paths from the left cell (`dp[col - 1]`) and keeps the count from the above cell (`dp[col]`).

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = 1
        
        for row in range(m):
            for col in range(n):
                if row == 0 and col == 0:
                    continue

                if row == 0:
                    dp[row][col] += dp[row][col-1]
                elif col == 0:
                    dp[row][col] += dp[row-1][col]
                else:
                    dp[row][col] += (dp[row][col-1] + dp[row-1][col])
        return dp[m-1][n-1]
        
```

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n  # Initialize a 1D dp array with 1s, representing the first row
        
        for row in range(1, m):  # Start from the second row
            for col in range(1, n):  # Start from the second column
                dp[col] += dp[col - 1]  # Update the current cell with the sum of the cell to the left and itself

        return dp[-1]  # The last element contains the number of unique paths

```

## Unique Paths II

**Solution**:

1. **Handle Edge Case for Starting Obstacle**:
   - If the starting cell `(0, 0)` contains an obstacle, immediately return `0` since no paths are available.

2. **Initialize the 1D DP Array**:
   - Use a 1D `dp` array of size `n` (number of columns), initialized with `0`s.
   - Set `dp[0]` to `1` to represent the starting cell, assuming it's not blocked.

3. **Iterate Through the Grid and Update Paths with Obstacles**:
   - For each cell `(row, col)` in the grid:
     - If `obstacleGrid[row][col] == 1`, set `dp[col] = 0`, indicating that cell is unreachable.
     - Otherwise, if `col > 0`, update `dp[col] += dp[col - 1]` to accumulate paths from the left cell.
   - This approach effectively incorporates obstacles by setting paths to `0` where obstacles exist.

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if obstacleGrid[0][0] == 1:
            return 0

        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0]*n for _ in range(m)]
        dp[0][0] = 1

        for row in range(m):
            for col in range(n):
                if obstacleGrid[row][col] == 1:
                    continue

                if col == 0 and row == 0:
                    continue

                if row == 0:
                    dp[row][col] += dp[row][col-1]
                elif col == 0:
                    dp[row][col] += dp[row-1][col]
                else:
                    dp[row][col] += (dp[row][col-1] + dp[row-1][col])

        return dp[-1][-1]

```

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if obstacleGrid[0][0] == 1:
            return 0

        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [0] * n
        dp[0] = 1  # Start point

        for row in range(m):
            for col in range(n):
                if obstacleGrid[row][col] == 1:
                    dp[col] = 0  # Set dp[col] to 0 if there's an obstacle
                elif col > 0:
                    dp[col] += dp[col - 1]  # Update dp[col] by adding paths from the left

        return dp[-1]  # The number of unique paths to the bottom-right corner

```

## Integer Break

**Solution**:

1. **Define the Problem in Terms of Subproblems**:
   - Given an integer `n`, the task is to break it into at least two positive integers such that their product is maximized.
   - Define `dp[i]` as the maximum product obtainable by breaking the integer `i`.

2. **Base Case Initialization**:
   - Set `dp[2] = 1`, as the maximum product obtainable by breaking `2` is `1 * 1 = 1`.
   - There is no need to initialize `dp[0]` or `dp[1]` for this problem, as they do not contribute to any solution.

3. **Recurrence Relation / State Transition**:
   - For each integer `i` from `3` to `n`, consider splitting it into two parts: `j` and `i - j`.
   - The maximum product is calculated by:
     
     $dp[i] = \max(dp[i], j \times (i - j), j \times dp[i - j])$
     
   - This relation considers both:
     - Breaking `(i - j)` further (using `dp[i - j]`).
     - Not breaking `(i - j)` further, simply taking the product `j * (i - j)`.

4. **Iteration over Possible Splits**:
   - Only iterate over values of `j` up to `i // 2` for each `i` since splitting `i` at `j` or `i - j` yields symmetric results.

5. **Final Solution**:
   - After populating the `dp` array up to `n`, `dp[n]` contains the maximum product obtainable by breaking `n`.


```python
class Solution:
    def integerBreak(self, n: int) -> int:
        # Initialize dp array to store the maximum product for each integer up to n
        dp = [0] * (n + 1)
        
        # Base case: breaking 2 yields the maximum product of 1 (1 + 1)
        dp[2] = 1

        # Start filling dp array from 3 to n
        for i in range(3, n + 1):
            # Try to split the number i into two parts j and (i - j)
            for j in range(1, i // 2 + 1):
                # Calculate the maximum product by either:
                # 1. Not breaking (i - j) further: j * (i - j)
                # 2. Breaking (i - j) further using dp[i - j]
                dp[i] = max(j * (i - j), j * dp[i - j], dp[i])

        # The last element in dp array, dp[n], is the answer
        return dp[n]

```

## Unique Binary Search Trees

**Solution**:

1. **Define the Problem in Terms of Subproblems**:
   - The goal is to count the number of unique Binary Search Trees (BSTs) that can be formed with `n` distinct nodes.
   - Define `dp[i]` as the number of unique BSTs that can be formed using `i` nodes.

2. **Base Case Initialization**:
   - `dp[0] = 1`: An empty tree (0 nodes) has one unique structure.
   - `dp[1] = 1`: A single-node tree has only one unique structure.
   - `dp[2] = 2`: Two nodes can form two unique BSTs, either left-rooted or right-rooted.

3. **Recurrence Relation / State Transition**:
   - For each `i` from `3` to `n`, compute `dp[i]` by considering each possible node as the root of the tree.
   - If `j` is chosen as the root, the left subtree has `j - 1` nodes and the right subtree has `i - j` nodes.
   - The recurrence relation is:
     \[
     dp[i] += dp[j - 1] \times dp[i - j]
     \]
   - This accounts for all unique combinations of left and right subtrees formed with `i` nodes.

4. **Iteration over Possible Root Choices**:
   - For each `i`, iterate through each node `j` (from `1` to `i`) to consider it as the root.
   - Multiply the number of unique BSTs on the left and right subtrees to account for all possible configurations.

5. **Final Result**:
   - After filling the `dp` array, `dp[n]` will contain the number of unique BSTs that can be formed with `n` nodes.

### Complexity Analysis
- **Time Complexity**: \(O(n^2)\), due to the nested loops over `i` and `j` (each up to `n`).
- **Space Complexity**: \(O(n)\), for the `dp` array of size `n + 1`.

```python
class Solution:
    def numTrees(self, n: int) -> int:
        # Handle the base case for n=1 directly
        if n == 1:
            return 1

        # Initialize dp array where dp[i] represents the number of unique BSTs with i nodes
        dp = [0] * (n + 1)
        
        # Base cases:
        dp[0] = 1  # Empty tree (0 nodes) has one unique structure
        dp[1] = 1  # One node has only one unique structure
        dp[2] = 2  # Two nodes can be arranged in two unique BST structures
        
        # Fill dp array for each number of nodes from 3 to n
        for i in range(3, n + 1):
            # Calculate dp[i] by summing the number of unique BSTs for each possible root position
            for j in range(1, i + 1):
                # dp[j-1] represents left subtree options, dp[i-j] represents right subtree options
                dp[i] += dp[j - 1] * dp[i - j]
        
        # The result is the number of unique BSTs that can be formed with n nodes
        return dp[n]
    
```

1. **Why `j-1` for the Left Subtree**
- If `j` is the root, all nodes less than `j` go to the left subtree.
- The values `{1, 2, ..., j-1}` are all less than `j`, so there are exactly `j-1` nodes in the left subtree.
- `dp[j-1]` therefore represents the number of unique BSTs that can be formed with these `j-1` nodes.

2. **Why `i-j` for the Right Subtree**
- Similarly, all nodes greater than `j` go to the right subtree.
- Since `i` is the total number of nodes and `j` is the root (with `j-1` nodes in the left subtree), the remaining nodes for the right subtree are `i - j`.
- The values `{j+1, j+2, ..., i}` are greater than `j`, so there are exactly `i-j` nodes in the right subtree.
- `dp[i-j]` represents the number of unique BSTs that can be formed with these `i-j` nodes.

### Example: Calculating `dp[3]`

To see how this works in practice, let’s calculate `dp[3]`, the number of unique BSTs that can be formed with nodes `1, 2, 3`:

1. **Initialize Base Cases**:
   - `dp[0] = 1`: An empty tree has one unique structure.
   - `dp[1] = 1`: A single-node tree has only one unique structure.
   - `dp[2] = 2`: With two nodes, there are two unique BSTs: left-rooted and right-rooted trees.

2. **Calculate `dp[3]` by Considering Each Node as the Root**:
   - For each node `j` (from `1` to `3`) as the root, calculate the possible BSTs for the left and right subtrees.

#### Case-by-Case Breakdown:
- **When `j = 1`**:
  - Left subtree has `j - 1 = 0` nodes, so `dp[0] = 1`.
  - Right subtree has `i - j = 3 - 1 = 2` nodes, so `dp[2] = 2`.
  - Total BSTs with `1` as root: `dp[0] * dp[2] = 1 * 2 = 2`.

- **When `j = 2`**:
  - Left subtree has `j - 1 = 1` node, so `dp[1] = 1`.
  - Right subtree has `i - j = 3 - 2 = 1` node, so `dp[1] = 1`.
  - Total BSTs with `2` as root: `dp[1] * dp[1] = 1 * 1 = 1`.

- **When `j = 3`**:
  - Left subtree has `j - 1 = 2` nodes, so `dp[2] = 2`.
  - Right subtree has `i - j = 3 - 3 = 0` nodes, so `dp[0] = 1`.
  - Total BSTs with `3` as root: `dp[2] * dp[0] = 2 * 1 = 2`.

3. **Sum the Results**:
   - The total number of unique BSTs with `3` nodes is:
    $$
     dp[3] = dp[0] \times dp[2] + dp[1] \times dp[1] + dp[2] \times dp[0] = 2 + 1 + 2 = 5
    $$
   - Thus, `dp[3] = 5`.

This gives us the number of unique BSTs for `n = 3` and illustrates why `j-1` represents the left subtree options and `i-j` represents the right subtree options.


## 0-1 Knapsack Problem

1. **Problem Setup**

    The knapsack problem aims to maximize the total value of items included in a knapsack with a fixed weight capacity. Each item has:
    - **Weight**: How much space it takes in the knapsack.
    - **Value**: The benefit or worth of including it.
    The goal is to pick a combination of items that maximizes total value without exceeding the knapsack's capacity.

2. **Dynamic Programming Table (DP Table)**
    - The code constructs a 2D DP table `dp`, where `dp[i][j]` represents the maximum value achievable with the first `i` items and a knapsack of capacity `j`.
    - This table helps store intermediate results, avoiding redundant calculations 

3. **Initialization**
    - The table is initialized such that for all rows, `dp[row][0] = 0`, meaning if the knapsack has zero capacity, the maximum value is `0`.
    - For the first item (first row), if the knapsack's capacity is at least the weight of the first item (`items[0][0]`), then `dp[0][col] = items[0][1]`. This represents the value of the first item, as it's the only item available at that capacity.

4. **Filling the DP Table**
    - For each subsequent item (`i` from `1` to `len(items) - 1`) and for each possible knapsack capacity (`j` from `1` to `capacity`):
        - **Excluding the item**: If the current item is excluded, then `dp[i][j] = dp[i-1][j]`, representing the best achievable value with previous items and the current capacity.
        - **Including the item**: If the current item is included (only if `j >= items[i][0]`), then the achievable value is `dp[i-1][j-items[i][0]] + items[i][1]`. Here:
        - `dp[i-1][j-items[i][0]]` is the best achievable value using previous items with the remaining capacity after including the current item.
        - `items[i][1]` is the value of the current item itself.
    
    $$
        dp[i][j] = \begin{cases} 
        0 & \text{if } i = 0 \text{ or } j = 0 \\
        dp[i-1][j] & \text{if } w_i > j \\
        \max(dp[i-1][j], dp[i-1][j - w_i] + v_i) & \text{if } w_i \leq j 
        \end{cases}
    $$

```python
from typing import List, Tuple

def knapsack(capacity: int, items: List[Tuple[int, int]]) -> int:
    # Initialize a 2D DP table with 0s, where dp[i][j] represents the maximum value achievable
    # with the first i items and knapsack capacity j.
    dp = [[0] * (capacity + 1) for _ in range(len(items))]

    # Set the base case for each item where knapsack capacity is 0 (no value can be carried).
    for row in range(len(items)):
        dp[row][0] = 0

    # Initialize the first row: If the capacity allows, we take the first item.
    for col in range(capacity + 1):
        if col >= items[0][0]:  # Check if the capacity can hold the first item's weight
            dp[0][col] = items[0][1]  # Set the value as the item's value

    # Fill the DP table for each item and capacity
    for i in range(1, len(items)):
        for j in range(1, capacity + 1):
            # If the current item's weight is less than or equal to the current capacity j
            if j >= items[i][0]:
                # Choose the maximum value between excluding and including the current item
                dp[i][j] = max(dp[i-1][j], dp[i-1][j - items[i][0]] + items[i][1])
            else:
                # Otherwise, we can't include the current item, so we take the previous value
                dp[i][j] = dp[i-1][j]

    # Print the DP table for debugging purposes to see the values at each step
    print(dp)
    
    # The solution is in the bottom-right cell, representing the maximum value for the full capacity
    return dp[-1][-1]

```

**Optimization**:

1. **1D DP Array Initialization**:
   - Instead of a 2D array, a 1D array `dp` is used where `dp[j]` holds the maximum value for a knapsack capacity `j`.
   - This reduces the space complexity from $( O(\text{n} \times \text{capacity}) )$ to $ ( O(\text{capacity}) )$.

```python
def knapsack_1D(capacity: int, items: List[Tuple[int, int]]) -> int:
    dp = [0] * (capacity+1)

    for capacity_index in range(capacity+1):
        if capacity_index >= items[0][0]:
            dp[capacity_index] = items[0][1]

    for item_index in range(1, len(items)):
        for capacity_index in range(1, capacity+1):
            if capacity_index >= items[item_index][0]:
                dp[capacity_index] = max(dp[capacity_index], dp[capacity_index-items[item_index][0]] + items[item_index][1])
        print(dp)

    return dp[-1]

```

## Partition Equal Subset Sum

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        target = sum(nums)
        
        # If the total sum is odd, it's impossible to split it into two equal subsets
        if target % 2 == 1:
            return False
        
        # Calculate the target sum for one subset (half of total sum)
        target //= 2
        
        # Initialize a 1D DP array where dp[j] represents the maximum achievable subset sum with sum j
        dp = [0] * (target + 1)
        
        # Process each number in the array
        for num in nums:
            # Update the DP array in reverse order to prevent using the same item multiple times
            for j in range(target, num - 1, -1):
                # Update dp[j] to be the maximum of:
                # - dp[j] (not including num)
                # - dp[j - num] + num (including num to achieve a new sum j)
                dp[j] = max(dp[j], dp[j - num] + num)

        # Check if the target sum is achievable; if dp[target] == target, we found an equal partition
        return dp[target] == target
        
```

## Last Stone Weight II

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        # If there's only one stone, return its weight since there's nothing to balance it against.
        if len(stones) == 1:
            return stones[0]
        
        # Calculate the target weight, which is half the total sum of the stones.
        # We aim to find the maximum possible subset sum close to this target.
        target = sum(stones) // 2
        
        # Initialize the dp array where dp[j] represents the closest subset sum we can achieve to 'j'.
        dp = [0] * (target + 1)

        # Iterate through each stone in the stones list
        for i in range(len(stones)):
            # Traverse backwards in the dp array to avoid reusing the same stone in this iteration
            for j in range(target, stones[i] - 1, -1):
                # Update dp[j] by choosing the maximum of:
                # - keeping the current value (not including this stone in the subset)
                # - adding the current stone to the subset (dp[j - stones[i]] + stones[i])
                dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])

        # After filling the dp array, dp[target] holds the best achievable subset sum closest to the target
        # The result is the minimum difference between two groups, calculated as:
        # (total sum of stones) - 2 * (best achievable subset sum close to half)
        return sum(stones) - dp[target] * 2

```

## Target Sum

**Time Limit Exceeded**: Using Backtracing

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # Initialize the result counter
        result = 0
        
        # Define the backtracking helper function
        def backtracking(index: int, currentSum: int):
            nonlocal result
            # Base case: all elements processed
            if index == len(nums):
                # Check if currentSum matches target
                if currentSum == target:
                    result += 1  # Found a valid way
                return
            
            # Recursive case: add the current element
            backtracking(index + 1, currentSum + nums[index])
            # Recursive case: subtract the current element
            backtracking(index + 1, currentSum - nums[index])
        
        # Start the backtracking process
        backtracking(0, 0)
        return result
    
```

### Problem Transformation

The problem of assigning `+` and `-` signs to elements in `nums` to achieve a `target` sum can be transformed into a **subset sum problem**. Here’s how:

1. Let’s define two subsets:
   - `P`: The subset of elements with a `+` sign.
   - `N`: The subset of elements with a `-` sign.
   
2. Based on these subsets, we have:
   - \( P - N = target)
   - \( P + N = sum(nums))
   
3. Solving for `P`, we get:
   - \( P = (sum(nums) + target ) / 2 \)
   
Thus, we need to find the number of ways to achieve a subset sum equal to \( subset\_sum = (sum(nums) + target ) / 2 \).

### Edge Cases

- **Odd Sum Check**: If `sum(nums) + target` is odd, then `subset_sum` is not an integer, and it’s impossible to partition `nums` to meet the target. We return `0`.
- **Target Out of Reach**: If \( `abs(target) > sum(nums)` \), the target is unachievable, so we return `0`.

### Dynamic Programming Array Setup

1. **Define `dp` Array**: 
   - `dp[j]` will store the number of ways to achieve the sum `j` using elements from `nums`.

2. **Initialize `dp[0]`**:
   - Set `dp[0] = 1` because there is one way to achieve a sum of `0` (by selecting no elements).

### DP Transition (Subset Sum Calculation)

- For each element in `nums`, update the `dp` array in reverse (to avoid reusing elements):
  - For each `j` from `subset_sum` down to `num`:
    - `dp[j] += dp[j - num]`
  - This step accumulates the number of ways to achieve each possible sum up to `subset_sum`.

### Result

- After processing all elements, `dp[subset_sum]` will hold the number of ways to reach `subset_sum`, which corresponds to the number of ways to assign signs to reach `target`.

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # Calculate the sum of all numbers
        total_sum = sum(nums)
        
        # Check if it’s possible to achieve the target
        # If total_sum + target is odd or if target is too large to be achieved, return 0
        if (total_sum + target) % 2 == 1 or total_sum < abs(target):
            return 0
        
        # Calculate the subset sum we need to achieve
        subset_sum = (total_sum + target) // 2
        
        # Initialize dp array, where dp[j] will store the number of ways to achieve sum j
        dp = [0] * (subset_sum + 1)
        dp[0] = 1  # There's one way to make zero sum: choose no elements
        
        # Fill the dp array by iterating through each number in nums
        for num in nums:
            # Traverse backwards to avoid reusing the same number in the same iteration
            for j in range(subset_sum, num - 1, -1):
                dp[j] += dp[j - num]
        
        # The answer is the number of ways to achieve the subset_sum
        return dp[subset_sum]
          
```

## Ones and Zeroes

Given a list of binary strings (`strs`), each string is composed of `0`s and `1`s. The goal is to determine the maximum number of strings that can be selected from `strs` such that the total count of `0`s is at most `m` and the total count of `1`s is at most `n`.

This problem can be framed as a **Knapsack problem**, where each string represents an item with:
- A "weight" in terms of the count of `0`s and `1`s.
- A "value" of `1` (since choosing the string increases the count by one).



1. **Define the DP Array**:
   - Let `dp[i][j]` represent the maximum number of strings that can be formed using at most `i` zeros and `j` ones.
   - Initialize a DP array with dimensions `(m + 1) x (n + 1)` where each cell starts with `0`:
     ```python
     dp = [[0] * (n + 1) for _ in range(m + 1)]
     ```

2. **Process Each String in `strs`**:
   - For each string in `strs`, count the number of `0`s and `1`s:
     ```python
     zeros = str.count('0')
     ones = str.count('1')
     ```
   - These counts (`zeros` and `ones`) represent the "cost" of choosing this string in terms of the `0` and `1` limits.

3. **Update the DP Array in Reverse**:
   - Use reverse iteration for each `i` (from `m` down to `zeros`) and `j` (from `n` down to `ones`) to avoid reusing the same string in the same iteration:
     ```python
     for i in range(m, zeros - 1, -1):
         for j in range(n, ones - 1, -1):
             dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
     ```
   - **Update Rule**: For each cell `dp[i][j]`, update it to the maximum of:
     - `dp[i][j]`: The current value, representing not picking this string.
     - `dp[i - zeros][j - ones] + 1`: The value after picking this string, which adds `1` to the previous state.

4. **Final Result**:
   - After processing all strings, the value at `dp[m][n]` will hold the maximum number of strings that can be formed using at most `m` zeros and `n` ones:
     ```python
     return dp[m][n]
     ```

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0] * (n+1) for _ in range(m+1)]
        
        for str in strs:
            zeros = str.count('0')
            ones = str.count('1')
            for i in range(m, zeros-1, -1):
                for j in range(n, ones-1, -1):
                    dp[i][j] = max(dp[i][j], dp[i-zeros][j-ones] + 1)


        return dp[m][n]

```

## Coin Change II

**Solution**:

1. **Define the DP Array**:
   - Let `dp[j]` represent the number of ways to achieve the amount `j` using the available coins.
   - Initialize `dp` with a size of `amount + 1`, where each entry is initially `0`, except for `dp[0]`, which is set to `1` (there is one way to make amount `0` by choosing no coins).

2. **Iterate Over Each Coin**:
   - For each `coin` in `coins`, update the `dp` array to reflect the new ways to achieve each amount by including that coin.

3. **Update the DP Array**:
   - For each amount `j` from `coin` up to `amount`, add the number of ways to achieve `j - coin` to `dp[j]`. This is because if we can achieve the amount `j - coin`, we can reach `j` by adding one more `coin`.
   - The update rule:
     ```python
     dp[j] += dp[j - coin]
     ```
   - This formula accumulates the ways to reach each amount `j` by considering the current coin's contribution.

4. **Final Result**:
   - After processing all coins, the value `dp[amount]` will contain the total number of ways to achieve the target `amount` using the available coins.

```python
lass Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        # Initialize a DP array where dp[j] represents the number of ways to achieve amount j
        dp = [0] * (amount + 1)
        dp[0] = 1  # There is one way to make amount 0: use no coins
        
        # Iterate over each coin in coins
        for coin in coins:
            # Update the dp array from the value of the coin to the target amount
            for j in range(coin, amount + 1):
                # Add the ways to achieve amount j - coin to dp[j]
                dp[j] += dp[j - coin]
        
        # The final answer is the number of ways to make up the 'amount'
        return dp[amount]

```

## Combination Sum IV

**Solution**:

1. **Define the DP Array**:
   - Let `dp[j]` represent the number of ways to achieve the sum `j` using the elements in `nums`.
   - Initialize `dp` with size `target + 1`, where each entry starts as `0`, except for `dp[0]`, which is set to `1` (there is one way to make the sum `0` by choosing no elements).

2. **Iterate Over Possible Sums**:
   - Loop through each possible sum `j` from `1` to `target`.
   - For each `j`, consider each element `num` in `nums` to determine if `num` can contribute to reaching `j`.

3. **Update the DP Array**:
   - For each sum `j` and each `num` in `nums`:
     - If `num` can be part of a combination to reach `j` (i.e., if `j >= num`):
       - Add the number of ways to achieve `j - num` to `dp[j]`.
       - This is because if we can reach `j - num`, then adding `num` would allow us to reach `j`.
     - The update rule:
       ```python
       dp[j] += dp[j - num]
       ```
   - This accumulates the ways to reach each possible sum by building upon smaller sums.

4. **Final Result**:
   - After processing all sums up to `target`, the value `dp[target]` will contain the total number of ways to achieve the target sum using the given numbers in `nums`.

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        # Initialize a DP array where dp[j] represents the number of ways to achieve amount j
        dp = [0] * (amount + 1)
        dp[0] = 1  # There is one way to make amount 0: use no coins
        
        # Iterate over each coin in coins
        for coin in coins:
            # Update the dp array from the value of the coin to the target amount
            for j in range(coin, amount + 1):
                # Add the ways to achieve amount j - coin to dp[j]
                dp[j] += dp[j - coin]
        
        # The final answer is the number of ways to make up the 'amount'
        return dp[amount]
    
```

## Coin Change

**Solution**:

1. **Define the DP Array**:
   - `dp[j]` will represent the minimum number of coins required to achieve amount `j`.
   - Initialize `dp` as `[float('inf')] * (amount + 1)` to represent initially unreachable amounts.
   - Set `dp[0] = 0` as the base case, since zero coins are needed to achieve amount `0`.

2. **Iterate Over Each Coin**:
   - For each coin in `coins`, update the `dp` array to reflect the minimum coins needed for amounts from `coin` up to `amount`.

3. **Update the DP Array**:
   - For each amount `j` (from `coin` to `amount`):
     - Update `dp[j]` to be the minimum of its current value or `dp[j - coin] + 1`.
     - This update rule means that if we can make the amount `j - coin`, then we can make `j` by adding one more coin of this denomination.

4. **Final Result**:
   - After processing all coins, check if `dp[amount]` is still `float('inf')`:
     - If it is, return `-1`, as it is impossible to form the amount with the given coins.
     - Otherwise, `dp[amount]` contains the minimum number of coins required to make up `amount`.


```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # Initialize the DP array with infinity, representing an initially unreachable amount
        dp = [float('inf')] * (amount + 1)
        
        # Base case: 0 coins are needed to make amount 0
        dp[0] = 0
        
        # Loop through each coin in the coins list
        for coin in coins:
            # For each coin, update the dp array for amounts from `coin` to `amount`
            for j in range(coin, amount + 1):
                # Update dp[j] to be the minimum of its current value or
                # the value of dp[j - coin] + 1 (adding this coin to the minimum solution for `j - coin`)
                dp[j] = min(dp[j], dp[j - coin] + 1)
        
        # If dp[amount] is still infinity, it means the amount cannot be made with the given coins
        if dp[amount] == float('inf'):
            return -1
        
        # Otherwise, return the minimum number of coins needed for `amount`
        return dp[amount]
    
```

## Perfect Squares

**Solution**:
1. **Define the DP Array**:
   - `dp[j]` will store the minimum number of perfect squares required to achieve the sum `j`.
   - Initialize `dp` as `[float('inf')] * (n + 1)`, with `dp[0] = 0` (no squares are needed to sum to `0`).

2. **Iterate Over Each Perfect Square**:
   - For each integer `i` from `1` up to `sqrt(n)`, calculate `i**2` as a perfect square.
   - This square represents a possible component for achieving sums from `i**2` up to `n`.

3. **Update the DP Array for Each Sum**:
   - For each target sum `j` (from `i**2` to `n`):
     - Update `dp[j]` to be the minimum of its current value or `dp[j - i**2] + 1`.
     - This formula means that if we can achieve `j - i**2`, then adding one instance of `i**2` will let us achieve `j` with one additional square.

4. **Final Result**:
   - After processing all perfect squares, `dp[n]` will contain the minimum number of squares required to sum to `n`.

```python
class Solution:
    def numSquares(self, n: int) -> int:
        # Initialize the DP array with infinity, representing an initially unreachable sum
        dp = [float('inf')] * (n + 1)
        
        # Base case: 0 perfect squares are needed to achieve sum 0
        dp[0] = 0
        
        # Iterate over each integer from 1 up to n
        for i in range(1, n + 1):
            # Calculate the square of i and update the dp array for amounts from i**2 up to n
            square = i ** 2
            for j in range(square, n + 1):
                # Update dp[j] to be the minimum of its current value or dp[j - square] + 1
                # dp[j - square] + 1 represents adding one perfect square (i^2) to the solution for (j - square)
                dp[j] = min(dp[j], dp[j - square] + 1)
        
        # The final answer is the minimum number of perfect squares needed for sum n
        return dp[n]
    
```

## Word Break

**Solution**:
1. **Define the DP Array**:
   - Let `dp[j]` represent whether the substring `s[:j]` can be segmented using words from `wordDict`.
   - Initialize `dp` with `False` values, except for `dp[0] = True` as a base case, since an empty string can always be segmented.

2. **Outer Loop: Iterate Over Each Position in `s`**:
   - For each position `j` from `1` to `len(s)`, check if the substring `s[:j]` can be formed by appending any word from `wordDict` to a previous valid substring.

3. **Inner Loop: Check Each Word in `wordDict`**:
   - For each word, check if:
     - The word can fit in the current substring length (`j >= len(word)`).
     - The substring `s[j - len(word):j]` matches the word.
     - If both conditions hold and `dp[j - len(word)]` is `True`, set `dp[j] = True` and break out of the loop, as a valid segmentation up to `j` has been found.

4. **Return the Result**:
   - After processing all positions, `dp[len(s)]` will indicate if the entire string `s` can be segmented using the words in `wordDict`.

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # Initialize the DP array where dp[j] is True if s[:j] can be segmented using words in wordDict
        dp = [False] * (len(s) + 1)
        dp[0] = True  # Base case: empty string can be segmented
        
        # Loop through each position in the string
        for j in range(1, len(s) + 1):
            # Check each word in wordDict to see if it can end at position j
            for word in wordDict:
                # Ensure the word length is not greater than j
                if j >= len(word):
                    # If dp[j - len(word)] is True and the substring matches the word, set dp[j] to True
                    if dp[j - len(word)] and s[j - len(word):j] == word:
                        dp[j] = True
                        break  # No need to check further if dp[j] is True
        
        # Return whether the entire string can be segmented
        return dp[-1]
        
```

## House Robber

**Solution**:
1. **Edge Case Handling**:
   - If there is only one house (`len(nums) == 1`), rob that house.
   - If there are two houses (`len(nums) == 2`), rob the house with the most money.

2. **Define the DP Array**:
   - Let `dp[i]` represent the maximum amount of money that can be robbed from the first `i+1` houses.
   - Initialize `dp[0] = nums[0]` since the only option is to rob the first house.
   - Initialize `dp[1] = max(nums[0], nums[1])` to rob the house with the larger amount from the first two houses.

3. **Fill the DP Array**:
   - For each house `i` from `2` to `len(nums) - 1`, calculate:
     - `dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])`
     - This formula means that `dp[i]` takes the maximum of either:
       - Not robbing house `i` (`dp[i - 1]`).
       - Robbing house `i`, which adds `nums[i]` to the best solution up to `i-2` (`dp[i - 2]`).

4. **Return the Result**:
   - The value `dp[-1]` (or `dp[len(nums) - 1]`) contains the maximum money that can be robbed without triggering an alarm.


```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]  # Only one house, so rob it
        if len(nums) == 2:
            return max(nums[0], nums[1])  # Two houses, pick the one with more money

        # Initialize DP array
        dp = [0] * len(nums)
        dp[0] = nums[0]  # Only one house, rob it
        dp[1] = max(nums[0], nums[1])  # Rob the house with more money

        # Fill the dp array for each house from the third onward
        for i in range(2, len(nums)):
            # Choose the max between not robbing current house or robbing it
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

        # The answer is the max amount that can be robbed from all houses
        return dp[-1]
    
```

## House Robber II

**Solution**:

1. **Two Scenarios**:
   - Since houses are in a circle, divide the problem into two scenarios:
     - **Exclude the last house**: Solve for houses from the first to the second-to-last.
     - **Exclude the first house**: Solve for houses from the second to the last.

## House Robber III

**Solution**:

In "House Robber III", houses are represented by nodes in a binary tree. Adjacent nodes (directly connected) cannot both be robbed, so the objective is to maximize the sum of non-adjacent nodes' values.

This problem can be solved using a **postorder traversal** of the tree and dynamic programming principles, where each node has two states:
   - **Rob this node**: Add its value and proceed without robbing its children.
   - **Do not rob this node**: Take the maximum possible money from each child node, whether they are robbed or not.

1. **Define the Traversal Function**:
   - Create a helper function `traversal` that returns two values for each node:
     - `dp[0]`: Maximum money if the node is **not robbed**.
     - `dp[1]`: Maximum money if the node **is robbed**.

2. **Recursive Calculations**:
   - For each node, use the results from the left and right children:
     - **`dp[0]` (not robbing this node)**: Sum of the maximum values from robbing or not robbing each child.
     - **`dp[1]` (robbing this node)**: Current node’s value plus the money from not robbing each child.

3. **Base Case**:
   - If a node is `None`, return `[0, 0]`, meaning there is no money to rob.

4. **Final Result**:
   - After the traversal, return `max(dp[0], dp[1])` for the root node, representing the maximum money that can be robbed with or without robbing the root.

```python
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        # Define a helper function for traversal that returns an array with two values:
        # - dp[0]: Maximum money if the current node is NOT robbed
        # - dp[1]: Maximum money if the current node IS robbed
        def traversal(cur: Optional[TreeNode]) -> List[int]:
            if not cur:
                # Base case: if the node is None, return [0, 0]
                return [0, 0]
            
            # Recursively solve for the left and right subtrees
            leftDp = traversal(cur.left)
            rightDp = traversal(cur.right)

            # dp[0]: If we do not rob this node, take the max of robbing or not robbing the children
            # dp[1]: If we rob this node, we cannot rob its children, so add its value to leftDp[0] and rightDp[0]
            return [
                max(leftDp[0], leftDp[1]) + max(rightDp[0], rightDp[1]),  # Max money without robbing this node
                cur.val + leftDp[0] + rightDp[0]  # Max money with robbing this node
            ]
        
        # Calculate the results for the root node
        dp = traversal(root)
        # Return the maximum money by choosing either to rob or not to rob the root node
        return max(dp[0], dp[1])
    
```

## Best Time to Buy and Sell Stock

**Solution**:

1. **Define the DP Array**:
   - `dp[i][0]`: Maximum profit on day `i` with **no stock held** at the end of the day.
   - `dp[i][1]`: Maximum profit on day `i` with **one stock held** at the end of the day.
   - Initialize `dp[0][0] = 0` and `dp[0][1] = -prices[0]`.

2. **Populate the DP Array**:
   - For each day `i`, update:
     - `dp[i][0]`: Maximum profit if **no stock is held** at the end of day `i`.
       - Choices: keep previous max without stock or sell stock today if it was bought on an earlier day.
       - Formula: `dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])`
     - `dp[i][1]`: Maximum profit if **one stock is held** at the end of day `i`.
       - Choices: keep previous max with stock or buy stock today (resetting to -prices[i] since only one transaction is allowed).
       - Formula: `dp[i][1] = max(dp[i-1][1], -prices[i])`

3. **Final Result**:
   - The maximum profit at the end of the last day, holding no stock, is `dp[-1][0]`.

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # Edge case: if there is only one day, no transactions can be made
        if len(prices) == 1:
            return 0
        
        # Initialize the DP array where dp[i][0] is the max profit on day i with no stock
        # and dp[i][1] is the max profit on day i with one stock (bought at some point)
        dp = [[0, 0] for _ in range(len(prices))]
        
        # Base cases
        dp[0][0] = 0           # If we do not buy on the first day, profit is 0
        dp[0][1] = -prices[0]  # If we buy on the first day, profit is -prices[0]
        
        # Fill the DP array for each day
        for i in range(1, len(prices)):
            # dp[i][0]: max profit if we do not hold stock on day i
            # Choices: not selling (keep dp[i-1][0]) or sell stock bought on an earlier day (dp[i-1][1] + prices[i])
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            
            # dp[i][1]: max profit if we hold stock on day i
            # Choices: keep holding (dp[i-1][1]) or buy stock today (since only one transaction is allowed, set to -prices[i])
            dp[i][1] = max(dp[i-1][1], -prices[i])

        # The maximum profit achievable is without holding stock on the last day (dp[-1][0])
        return dp[-1][0]

```

## Best Time to Buy and Sell Stock II

**Solution**:

1. **Define the DP Array**:
   - `dp[i][0]`: Maximum profit up to day `i` with no stock held.
   - `dp[i][1]`: Maximum profit up to day `i` with stock held.
   - Initialize `dp[0][0] = 0` (no profit if no transactions on the first day).
   - Initialize `dp[0][1] = -prices[0]` (negative profit if stock bought on the first day).

2. **Fill the DP Array**:
   - For each day `i`, calculate:
     - **`dp[i][0]`**: Maximum profit if no stock is held on day `i`.
       - Choices:
         - Do nothing: `dp[i-1][0]` (same profit as the previous day).
         - Sell stock: `dp[i-1][1] + prices[i]` (profit from selling stock bought on a previous day).
       - Formula: `dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])`
     - **`dp[i][1]`**: Maximum profit if stock is held on day `i`.
       - Choices:
         - Do nothing: `dp[i-1][1]` (same as previous day's profit with stock).
         - Buy stock: `dp[i-1][0] - prices[i]` (profit from buying stock with previous day’s `no stock` profit).
       - Formula: `dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])`

3. **Final Result**:
   - The maximum profit at the end of the last day with no stock held is `dp[-1][0]`.

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # Edge case: if there's only one day, no transactions can be made
        if len(prices) == 1:
            return 0
        
        # Initialize the DP array:
        # dp[i][0]: max profit on day i with no stock held
        # dp[i][1]: max profit on day i with stock held
        dp = [[0, 0] for _ in range(len(prices))]
        
        # Base cases
        dp[0][0] = 0               # No stock on the first day means no profit
        dp[0][1] = -prices[0]      # Buying stock on the first day costs prices[0]
        
        # Fill in the DP array for each subsequent day
        for i in range(1, len(prices)):
            # dp[i][0]: max profit if no stock is held on day i
            # Choices: do nothing (dp[i-1][0]) or sell stock bought on an earlier day (dp[i-1][1] + prices[i])
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            
            # dp[i][1]: max profit if stock is held on day i
            # Choices: keep holding stock (dp[i-1][1]) or buy stock today (dp[i-1][0] - prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])

        # Return the max profit at the end of the last day with no stock held
        return dp[-1][0]
        
```

## Best Time to Buy and Sell Stock III

**Solution**:

1. **Define the DP Array**:
   - Initialize `dp[i][0]` to track the max profit with no transactions.
   - Initialize `dp[i][1]` to track the max profit with the first buy.
   - Initialize `dp[i][2]` to track the max profit with the first transaction completed.
   - Initialize `dp[i][3]` to track the max profit with the second buy done.

2. **Base Case Initialization**:
   - `dp[0][0] = 0`: No profit without any transaction.
   - `dp[0][1] = -prices[0]`: Cost of buying stock on the first day for the first transaction.
   - `dp[0][2] = 0`: No profit with no completed transactions on the first day.
   - `dp[0][3] = -prices[0]`: Cost of buying stock on the first day for the second transaction.

3. **Iterate Through Days**:
   - For each day `i`, update each state:
     - **`dp[i][0]`**: Max profit with no transactions.
     - **`dp[i][1]`**: Max profit with the first buy.
     - **`dp[i][2]`**: Max profit with the second transaction completed.
     - **`dp[i][3]`**: Max profit with the second buy.

4. **Final Result**:
   - The maximum profit achievable with up to two transactions is `dp[-1][2]`.

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # Edge case: If only one price is given, return 0 as no transactions can be made
        if len(prices) == 1:
            return 0
        
        # Initialize DP array:
        # dp[i][0]: max profit with first no holding stock up to day i
        # dp[i][1]: max profit with first holding stock up to day i
        # dp[i][2]: max profit with second no holding stock to day i
        # dp[i][3]: max profit with second holding stock up to day i
        dp = [[0, 0, 0, 0] for _ in range(len(prices))]
        dp[0][0], dp[0][1], dp[0][2], dp[0][3] = 0, -prices[0], 0, -prices[0]

        # Populate the DP array for each day
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], -prices[i])
            dp[i][2] = max(dp[i-1][2], dp[i-1][3] + prices[i])
            dp[i][3] = max(dp[i-1][3], dp[i-1][0] - prices[i])

        # Return the maximum profit after completing up to two transactions (dp[-1][2])
        return dp[-1][2]
    
```

## Best Time to Buy and Sell Stock IV

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if len(prices) == 1:
            return 0
        
        # Initialize DP array with 2*k states for each day, to handle multiple transactions
        # dp[i][j]: max profit on day i with a specific transaction state j
        dp = [[0] * (2 * k)  for _ in range(len(prices))]
        for i in range(2 * k):
            if i % 2 == 0:
                dp[0][i] = 0
            else:
                dp[0][i] = -prices[0]

        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], -prices[i])

            for j in range(2, 2 * k, 2):
                dp[i][j] = max(dp[i-1][j], dp[i-1][j+1] + prices[i])
                dp[i][j+1] = max(dp[i-1][j+1], dp[i-1][j+1-3] - prices[i])

        return dp[-1][-2]

```

## Best Time to Buy and Sell Stock with Cooldown

**Solution**:


```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # Edge case: if there is only one price, no transactions can be made
        if len(prices) == 1:
            return 0
        
        # Initialize DP array where:
        # dp[i][0] represents the max profit on day i without holding any stock.
        # dp[i][1] represents the max profit on day i while holding a stock.
        dp = [[0, 0] for _ in range(len(prices))]
        
        # Base cases
        dp[0][0] = 0            # No stock held on day 0, profit is 0
        dp[0][1] = -prices[0]    # Stock bought on day 0, negative profit by price of the stock
        dp[1][0] = max(dp[0][0], dp[0][1] + prices[1])  # Sell the stock on day 1 or do nothing
        dp[1][1] = max(dp[0][1], -prices[1])            # Buy the stock on day 1 or hold from day 0
        
        # Populate the DP array for each day from day 2 onwards
        for i in range(2, len(prices)):
            # dp[i][0]: Max profit on day i without holding any stock
            # Choices: do nothing (dp[i-1][0]) or sell stock held from previous day (dp[i-1][1] + prices[i])
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            
            # dp[i][1]: Max profit on day i while holding a stock
            # Choices: keep holding (dp[i-1][1]) or buy stock today (dp[i-2][0] - prices[i] due to cooldown)
            dp[i][1] = max(dp[i-1][1], dp[i-2][0] - prices[i])

        # Return the max profit on the last day without holding any stock
        return dp[-1][0]
    
```

## Best Time to Buy and Sell Stock with Transaction Fee

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        # Edge case: if only one price is given, no transactions can be made
        if len(prices) == 1:
            return 0
        
        # Initialize DP array where:
        # dp[i][0] represents the max profit on day i without holding any stock.
        # dp[i][1] represents the max profit on day i while holding a stock.
        dp = [[0, 0] for _ in range(len(prices))]
        
        # Base cases
        dp[0][0] = 0            # No stock held on day 0, profit is 0
        dp[0][1] = -prices[0]    # Stock bought on day 0, initial negative profit
        
        # Populate DP array for each subsequent day
        for i in range(1, len(prices)):
            # dp[i][0]: Max profit on day i without holding any stock
            # Choices: do nothing (dp[i-1][0]) or sell stock (dp[i-1][1] + prices[i] - fee)
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i] - fee)
            
            # dp[i][1]: Max profit on day i while holding a stock
            # Choices: keep holding (dp[i-1][1]) or buy stock (dp[i-1][0] - prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])

        # The maximum profit achievable by the last day without holding stock is in dp[-1][0]
        return dp[-1][0]
    
```

## Longest Increasing Subsequence

**Solution**:
1. **Define the DP Array**:
   - `dp[i]` represents the length of the longest increasing subsequence ending at index `i`.
   - Initialize each `dp[i]` as `1` since the minimum LIS ending at any element is the element itself.

2. **Populate the DP Array**:
   - For each element `i`, look at all previous elements `j` (from `0` to `i-1`):
     - If `nums[i] > nums[j]`, then `nums[i]` can extend the subsequence ending at `j`.
     - Update `dp[i]` as `max(dp[i], dp[j] + 1)` to reflect the longest sequence including `nums[i]`.

3. **Final Result**:
   - The length of the longest increasing subsequence is the maximum value in `dp`, representing the longest subsequence ending at each position.

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # Edge case: if there is only one element, the longest increasing subsequence is itself
        if len(nums) == 1:
            return 1
        
        # Initialize DP array where dp[i] represents the length of the longest increasing
        # subsequence ending at index i
        dp = [1] * len(nums)

        # Fill the DP array
        for i in range(1, len(nums)):
            for j in range(i):
                # If nums[i] can extend the increasing subsequence ending at j
                if nums[i] > nums[j]:
                    # Update dp[i] to the maximum of its current value or dp[j] + 1
                    dp[i] = max(dp[i], dp[j] + 1)
        
        # The longest increasing subsequence is the maximum value in dp array
        return max(dp)


"""
Replacing elements in lis ensures it remains as "low" as possible because 
smaller elements allow more room for subsequent elements to extend the subsequence
"""

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        def binarySearch(lis: List[int], target: int) -> int:
            """Find the position to insert target in lis using binary search."""
            left, right = 0, len(lis) - 1
            while left <= right:
                mid = (left + right) // 2
                if lis[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return left  # Position to insert target

        lis = []  # To store the smallest end elements of increasing subsequences
        for num in nums:
            pos = binarySearch(lis, num)
            if pos == len(lis):  # Extend lis if target is larger than all elements
                lis.append(num)
            else:  # Replace the element at pos to maintain optimal lis
                lis[pos] = num
        return len(lis)  # Length of lis is the length of LIS
        
```

## Longest Continuous Increasing Subsequence

**Solution**:

1. **Define the DP Array**:
   - `dp[i]` represents the length of the LCIS ending at index `i`.
   - Initialize `dp[i] = 1` for all indices since the minimum LCIS length ending at any element is `1`.

2. **Update the DP Array**:
   - For each index `i`:
     - If `nums[i] > nums[i-1]`, extend the LCIS: `dp[i] = dp[i-1] + 1`.
     - Otherwise, reset the LCIS at `i`: `dp[i] = 1`.

3. **Final Result**:
   - The length of the longest LCIS is `max(dp)`.

```python
class Solution:
     def findLengthOfLCIS(self, nums: List[int]) -> int:
        # Edge case: If the array has only one element, the LCIS is 1
        if len(nums) == 1:
            return 1
        
        # Initialize a DP array to track the LCIS ending at each index
        dp = [1] * len(nums)  # Each element starts as 1 since the minimum LCIS is the element itself
        
        # Iterate through the array starting from the second element
        for i in range(1, len(nums)):
            # If the current element is greater than the previous one, it extends the LCIS
            if nums[i] > nums[i-1]:
                dp[i] = dp[i-1] + 1  # Extend the previous LCIS length
        
        # The result is the maximum value in the DP array
        return max(dp)
     
```

## Maximum Length of Repeated Subarray

**Solution**:
1. **Define the DP Table**:
   - `dp[i][j]` represents the length of the longest common subarray ending at `nums1[i]` and `nums2[j]`.

2. **Initialization**:
   - For the first row and column:
     - If `nums1[i] == nums2[0]`, set `dp[i][0] = 1`.
     - If `nums2[j] == nums1[0]`, set `dp[0][j] = 1`.

3. **Update the DP Table**:
   - For each pair `(i, j)`:
     - If `nums1[i] == nums2[j]`, extend the subarray:
       - `dp[i][j] = dp[i-1][j-1] + 1`.
     - Otherwise, reset `dp[i][j] = 0`.

4. **Track Maximum Length**:
   - Use a variable `result` to track the maximum value in the DP table.

5. **Return Result**:
   - The final value of `result` is the maximum length of the repeated subarray.


```python
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        # Initialize the DP table
        # dp[i][j] will store the length of the longest common subarray ending at nums1[i] and nums2[j]
        dp = [[0] * len(nums2) for _ in range(len(nums1))]

        result = 0  # Variable to track the maximum length of repeated subarray
        
        # Fill the first column of the DP table
        for i in range(len(nums1)):
            dp[i][0] = 1 if nums1[i] == nums2[0] else 0
            result = max(result, dp[i][0])  # Update the maximum result
        
        # Fill the first row of the DP table
        for j in range(len(nums2)):
            dp[0][j] = 1 if nums2[j] == nums1[0] else 0
            result = max(result, dp[0][j])  # Update the maximum result

        # Populate the rest of the DP table
        for i in range(1, len(nums1)):
            for j in range(1, len(nums2)):
                if nums1[i] == nums2[j]:  # If characters match, extend the common subarray
                    dp[i][j] = dp[i-1][j-1] + 1
                    result = max(result, dp[i][j])  # Update the maximum result

        return result  # Return the maximum length of the repeated subarray
    
```

## Longest Common Subsequence

**Solution**:
1. **Define the DP Table**:
   - `dp[i][j]` represents the length of the LCS of `text1[:i]` and `text2[:j]`.

2. **Initialization**:
   - Base cases:
     - `dp[0][j] = 0` for all `j`: An empty string has an LCS of 0 with any string.
     - `dp[i][0] = 0` for all `i`.

3. **Update the DP Table**:
   - For each pair `(i, j)`:
     - If `text1[i-1] == text2[j-1]`, extend the LCS:
       - `dp[i][j] = dp[i-1][j-1] + 1`.
     - Otherwise, take the maximum LCS of excluding one character:
       - `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`.

4. **Result**:
   - The length of the LCS is stored in `dp[len(text1)][len(text2)]`.


```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # Initialize DP table with an extra row and column for the base case (0-indexed)
        # dp[i][j] represents the length of the LCS of text1[:i] and text2[:j]
        dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]

        # Variable to track the result (maximum length of the LCS)
        result = 0

        # Populate the DP table
        for i in range(1, len(text1) + 1):
            for j in range(1, len(text2) + 1):
                if text1[i-1] == text2[j-1]:  # If the characters match
                    dp[i][j] = dp[i-1][j-1] + 1  # Extend the LCS
                    result = max(result, dp[i][j])  # Update the result if needed
                else:
                    # Otherwise, take the maximum LCS without one of the characters
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        # Return the final LCS length
        return result
```

## Uncrossed Lines

**Solution**:

This problem is a variation of the **Longest Common Subsequence (LCS)** problem.

1. **Define the DP Table**:
   - `dp[i][j]` represents the maximum number of uncrossed lines between `nums1[:i]` and `nums2[:j]`.

2. **Update the DP Table**:
   - For each pair `(i, j)`:
     - If `nums1[i-1] == nums2[j-1]`, connect the elements and extend the count:
       - `dp[i][j] = dp[i-1][j-1] + 1`.
     - Otherwise, take the maximum from excluding one of the elements:
       - `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`.

3. **Final Result**:
   - The value `dp[len(nums1)][len(nums2)]` contains the maximum number of uncrossed lines.



```python
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        # Initialize DP table with dimensions (len(nums1)+1) x (len(nums2)+1)
        # dp[i][j] represents the maximum number of uncrossed lines between nums1[:i] and nums2[:j]
        dp = [[0] * (len(nums2) + 1) for _ in range(len(nums1) + 1)]

        result = 0  # Variable to store the maximum number of uncrossed lines

        # Fill the DP table
        for i in range(1, len(nums1) + 1):
            for j in range(1, len(nums2) + 1):
                if nums1[i-1] == nums2[j-1]:  # If the elements match
                    dp[i][j] = dp[i-1][j-1] + 1  # Extend the matching pair
                    result = max(result, dp[i][j])  # Update the result
                else:
                    # Otherwise, carry forward the maximum value by excluding one of the elements
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        # Return the maximum number of uncrossed lines
        return result

```

## Maximum Subarray

**Solution**:

1. **Define the DP Array**:
   - `dp[i]` represents the maximum sum of a subarray ending at index `i`.

2. **Recurrence Relation**:
   - At each index `i`:
     - Either extend the previous subarray: `dp[i-1] + nums[i]`.
     - Or start a new subarray at index `i`: `nums[i]`.
   - Formula: `dp[i] = max(nums[i], dp[i-1] + nums[i])`.

3. **Result**:
   - The maximum subarray sum is the largest value in `dp`: `max(dp)`.

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # Edge case: if there's only one element, the max subarray is the element itself
        if len(nums) == 1:
            return nums[0]
        
        # Initialize the DP array where dp[i] represents the maximum subarray sum ending at index i
        dp = [0] * len(nums)
        dp[0] = nums[0]  # Base case: max subarray sum at index 0 is nums[0]

        # Fill the DP array
        for i in range(1, len(nums)):
            # Either extend the previous subarray or start a new subarray at i
            dp[i] = max(nums[i], dp[i-1] + nums[i])

        # The result is the maximum value in the DP array
        return max(dp)
    
```
## Is Subsequence

1. Define a DP table `dp[i][j]`:
   - `dp[i][j]` represents the length of the subsequence of `s[:i]` in `t[:j]`.
2. Transition:
   - If `s[i-1] == t[j-1]`, extend the subsequence: `dp[i][j] = dp[i-1][j-1] + 1`.
   - Otherwise, carry forward the previous value: `dp[i][j] = dp[i][j-1]`.
3. Check:
   - If `dp[len(s)][len(t)] == len(s)`, then `s` is a subsequence of `t`.

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        if len(s) > len(t):
            return False
        
        dp = [[0] * (len(t) + 1) for _ in range(len(s) + 1)]

        for i in range(1, len(s) + 1):
            for j in range(1, len(t) + 1):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = dp[i][j-1]

        return dp[-1][-1] == len(s)
        
```

Two Points

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i, j = 0, 0  # Pointers for s and t

        # Traverse the string t
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1  # Move both pointers if characters match
            j += 1  # Always move pointer for t
        
        # If we've matched all characters of s, return True
        return i == len(s)

```

## Distinct Subsequences

**Solution**:

1. **Define the DP Table**:
   - `dp[i][j]` represents the number of ways to form `t[:j]` as a subsequence of `s[:i]`.

2. **Base Cases**:
   - `dp[i][0] = 1` for all `i`: One way to form an empty `t` (delete all characters in `s`).
   - `dp[0][j] = 0` for all `j > 0`: Impossible to form a non-empty `t` from an empty `s`.

3. **Recurrence Relation**:
   - If `s[i-1] == t[j-1]`:
     - Use the match or skip the character in `s`:
       - `dp[i][j] = dp[i-1][j-1] + dp[i-1][j]`.
   - If `s[i-1] != t[j-1]`:
     - Skip the character in `s`:
       - `dp[i][j] = dp[i-1][j]`.

4. **Result**:
   - The final value is stored in `dp[len(s)][len(t)]`.

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        # If s is shorter than t, it's impossible to form t
        if len(s) < len(t):
            return 0
        
        # Initialize DP table with dimensions (len(s)+1) x (len(t)+1)
        # dp[i][j] represents the number of ways to form t[:j] as a subsequence of s[:i]
        dp = [[0] * (len(t) + 1) for _ in range(len(s) + 1)]
        
        # Base case: An empty string t can be formed by deleting all characters of s
        for i in range(len(s) + 1):
            dp[i][0] = 1

        # Fill the DP table
        for i in range(1, len(s) + 1):
            for j in range(1, len(t) + 1):
                if s[i-1] == t[j-1]:  # Characters match
                    # Option 1: Use this match
                    # Option 2: Skip this character in s
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
                else:  # Characters don't match
                    # Skip this character in s
                    dp[i][j] = dp[i-1][j]

        # The result is stored in dp[len(s)][len(t)]
        return dp[-1][-1]
    
```
### Example: `s = "babgbag"` and `t = "bag"`

#### Step-by-Step Walkthrough

At `i = 4` (`s[:4] = "babg"`) and `j = 3` (`t[:3] = "bag"`):
- `s[3] = "g"` and `t[2] = "g"`. Characters **match**.
- When characters match, there are two choices:
  1. **Use this match**:
     - Use `s[3]` to match `t[2]`. In this case, the problem reduces to finding ways to form `"ba"` (`t[:2]`) from `"bab"` (`s[:3]`).
     - This value is stored in `dp[3][2]`.
  2. **Skip this match**:
     - Do not use `s[3]` to match `t[2]`. The problem remains finding ways to form `"bag"` (`t[:3]`) from `"bab"` (`s[:3]`).
     - This value is stored in `dp[3][3]`.


## 

**Solution**:

1. **Define the DP Table**:
   - `dp[i][j]` represents the minimum number of deletions needed to make `word1[:i]` and `word2[:j]` identical.

2. **Base Cases**:
   - If one string is empty:
     - Delete all characters of the other string.

3. **Recursive Relation**:
   - If `word1[i-1] == word2[j-1]`:
     - No deletion is needed:
       ```
       dp[i][j] = dp[i-1][j-1]
       ```
   - Otherwise:
     - Minimum deletions of three cases:
       1. Delete from `word1`.
       2. Delete from `word2`.
       3. Delete from both:
       ```
       dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 2)
       ```

4. **Final Result**:
   - The value in `dp[len(word1)][len(word2)]` gives the minimum deletions.

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # Initialize a DP table with dimensions (len(word1)+1) x (len(word2)+1)
        # dp[i][j] will store the minimum number of deletions required
        # to make word1[:i] and word2[:j] identical
        dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]

        # Base case 1: If one string is empty, the only option is to delete all characters
        # from the other string
        for i in range(len(word1) + 1):
            dp[i][0] = i  # Deleting all characters from word1[:i]
        for j in range(len(word2) + 1):
            dp[0][j] = j  # Deleting all characters from word2[:j]

        # Fill the DP table
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                # If the characters match, no additional deletion is required for these characters
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # Take the result from the previous diagonal cell
                else:
                    # If the characters do not match, consider three possible options:
                    # 1. Delete the character from word1: dp[i-1][j] + 1
                    # 2. Delete the character from word2: dp[i][j-1] + 1
                    # 3. Delete the characters from both word1 and word2: dp[i-1][j-1] + 2
                    dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 2)

        # The result is stored in the bottom-right corner of the DP table
        return dp[-1][-1]

```

## Edit Distance

**Solution**:

1. **Define the DP Table**:
   - `dp[i][j]` represents the minimum number of operations required to convert `word1[:i]` (first `i` characters of `word1`) into `word2[:j]` (first `j` characters of `word2`).

2. **Base Cases**:
   - If `word2` is empty, all characters of `word1` need to be deleted:
     ```
     dp[i][0] = i
     ```
   - If `word1` is empty, all characters of `word2` need to be inserted:
     ```
     dp[0][j] = j
     ```

3. **Recursive Relation**:
   - If `word1[i-1] == word2[j-1]`:
     - No operation is needed for this character:
       ```
       dp[i][j] = dp[i-1][j-1]
       ```
   - If `word1[i-1] != word2[j-1]`:
     - Consider the minimum cost of three operations:
       1. **Insert** a character into `word1`:
          ```
          dp[i][j] = dp[i][j-1] + 1
          ```
       2. **Delete** a character from `word1`:
          ```
          dp[i][j] = dp[i-1][j] + 1
          ```
       3. **Replace** a character in `word1`:
          ```
          dp[i][j] = dp[i-1][j-1] + 1
          ```

   - Combine these operations:
     ```
     dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)
     ```

4. **Result**:
   - The value in `dp[len(word1)][len(word2)]` gives the minimum edit distance.

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # Handle edge cases where one or both strings are empty
        if len(word1) == 0 or len(word2) == 0:
            # If either string is empty, the answer is the length of the other string
            return len(word1) if len(word2) == 0 else len(word2)
        
        # Initialize a DP table with dimensions (len(word1)+1) x (len(word2)+1)
        # dp[i][j] will represent the minimum number of operations required
        # to convert word1[:i] to word2[:j]
        dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        
        # Base case 1: If word2 is empty, delete all characters from word1
        for i in range(len(word1) + 1):
            dp[i][0] = i
        
        # Base case 2: If word1 is empty, insert all characters from word2
        for j in range(len(word2) + 1):
            dp[0][j] = j

        # Fill the DP table
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                # If the current characters match, no operation is needed for these characters
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # If characters do not match, consider three possible operations:
                    # 1. Delete from word1: dp[i-1][j] + 1
                    # 2. Insert into word1 (Delete from word2): dp[i][j-1] + 1
                    # 3. Replace the character: dp[i-1][j-1] + 1
                    dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)

        # The result is stored in dp[len(word1)][len(word2)]
        return dp[-1][-1]

```

## 

**Solution**:

1. **Define the DP Table**:
   - `dp[i][j]` is a boolean value that represents whether the substring `s[i:j+1]` is a palindrome.

2. **Base Cases**:
   - **Single-character substrings** (`i == j`): These are always palindromes.
   - **Two-character substrings** (`j - i == 1`): These are palindromes if the two characters are equal.

3. **Recursive Relation**:
   - For longer substrings (`j - i > 1`):
     - `s[i:j+1]` is a palindrome if:
       - The characters at `i` and `j` are the same (`s[i] == s[j]`), and
       - The inner substring `s[i+1:j]` is also a palindrome (`dp[i+1][j-1]`).

4. **Count the Palindromes**:
   - If `dp[i][j]` is `True`, increment the count of palindromic substrings.

5. **Result**:
   - The total number of palindromic substrings is stored in the variable `result`.


```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        # Edge case: If the string has only one character, there is exactly one palindrome
        if len(s) == 1:
            return 1

        # Initialize a 2D DP table where dp[i][j] indicates whether s[i:j+1] is a palindrome
        dp = [[False] * len(s) for _ in range(len(s))]

        result = 0  # Variable to store the count of palindromic substrings

        # Fill the DP table from bottom-right to top-left
        for i in range(len(s) - 1, -1, -1):  # Start from the last character of s
            for j in range(i, len(s)):  # For each character from i to the end of the string
                # Check if characters at i and j are the same
                if s[i] == s[j]:
                    # If the substring is of length 1 or 2, it's a palindrome
                    if j - i <= 1:
                        dp[i][j] = True
                        result += 1  # Increment count for this palindrome
                    # If it's longer, check the inner substring (dp[i+1][j-1])
                    elif dp[i+1][j-1]:
                        dp[i][j] = True
                        result += 1  # Increment count for this palindrome

        # Return the total count of palindromic substrings
        return result

```

Two Pointers:

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        def expandAroundCenter(left: int, right: int) -> int:
            count = 0
            # Expand outward while the characters match and stay within bounds
            while left >= 0 and right < len(s) and s[left] == s[right]:
                count += 1  # Found a palindrome
                left -= 1   # Move left pointer outward
                right += 1  # Move right pointer outward
            return count
        
        result = 0
        for i in range(len(s)):
            # Count odd-length palindromes (single character as center)
            result += expandAroundCenter(i, i)
            # Count even-length palindromes (two characters as center)
            result += expandAroundCenter(i, i + 1)
        
        return result
    
```

## Longest Palindromic Subsequence

**Solution**:
1. **Define the DP Table**:
   - `dp[i][j]` represents the length of the longest palindromic subsequence in `s[i:j+1]`.

2. **Base Case**:
   - A single character is always a palindrome:
     ```
     dp[i][i] = 1
     ```

3. **Recursive Relation**:
   - If `s[i] == s[j]`:
     - Extend the palindromic subsequence:
       ```
       dp[i][j] = dp[i+1][j-1] + 2
       ```
   - If `s[i] != s[j]`:
     - Exclude one character and take the maximum:
       ```
       dp[i][j] = max(dp[i+1][j], dp[i][j-1])
       ```

4. **Iterative Filling**:
   - Fill the DP table from the bottom-right to the top-left, ensuring all dependencies are computed.

5. **Result**:
   - The value in `dp[0][len(s)-1]` gives the length of the longest palindromic subsequence for the entire string.

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        # Edge case: If the string length is 1, the longest palindromic subsequence is the string itself
        if len(s) == 1:
            return 1
        
        # Initialize a DP table with dimensions (len(s) x len(s))
        # dp[i][j] represents the length of the longest palindromic subsequence in s[i:j+1]
        dp = [[0] * len(s) for _ in range(len(s))]

        # Fill the DP table
        # Iterate from the end of the string to the beginning (bottom-up approach)
        for i in range(len(s) - 1, -1, -1):  # Corrected range for the outer loop
            dp[i][i] = 1  # A single character is always a palindrome of length 1
            for j in range(i + 1, len(s)):  # j > i to avoid redundant computations
                if s[i] == s[j]:
                    # If the characters match, extend the palindromic subsequence
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    # If they don't match, take the max of excluding one character
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])

        # The result is stored in dp[0][-1], which considers the entire string
        return dp[0][-1]

```

# Monotonic Stack

A **monotonic stack** is a stack that maintains a specific order (either increasing or decreasing) among its elements. It is particularly useful for solving problems involving comparisons between elements, such as finding the **next greater element**, **next smaller element**, or solving **range queries**.

---

## Key Characteristics

1. **Monotonic Increasing Stack**:
   - Maintains elements in **increasing order** from bottom to top.
   - The top of the stack is the **smallest element** among all elements in the stack.

2. **Monotonic Decreasing Stack**:
   - Maintains elements in **decreasing order** from bottom to top.
   - The top of the stack is the **largest element** among all elements in the stack.


## Why Use a Monotonic Stack?

1. **Efficient Comparisons**:
   - Avoids brute force comparisons by keeping track of candidates for the next greater or smaller elements.
   - Reduces time complexity to \(O(n)\) in many cases.

2. **Applicable Problems**:
   - Finding **next greater/smaller elements**.
   - Solving **range queries** efficiently (e.g., sliding window maximum/minimum).
   - Problems involving **temperatures**, **stock spans**, or **histogram areas**.

---

## Daily Temperatures

1. Use a **monotonic decreasing stack** to keep track of indices of temperatures.
   - The stack ensures that for each index `i`, the temperatures corresponding to indices in the stack are in a strictly decreasing order.

2. For each temperature:
   - If the current temperature is greater than the temperature corresponding to the top of the stack:
     - Pop indices from the stack and compute the number of days until a warmer temperature.
   - Otherwise, push the current index onto the stack.

3. After traversing the temperatures:
   - Any indices left in the stack correspond to days with no warmer temperatures, so their `answer` remains `0`.

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        if len(temperatures) == 1:
            return [0]  # Single day means no warmer days ahead
        
        stack = []  # Monotonic stack to keep track of indices of decreasing temperatures
        answer = [0] * len(temperatures)  # Result array initialized with 0

        for i in range(len(temperatures)):
            # While the stack is not empty and the current temperature is greater
            # than the temperature at the index stored at the top of the stack
            while stack and temperatures[i] > temperatures[stack[-1]]:
                # Get the index of the last temperature that is smaller
                prev_index = stack.pop()
                # Calculate the number of days until a warmer temperature
                answer[prev_index] = i - prev_index
            
            # Push the current day's index onto the stack
            stack.append(i)

        # Indices left in the stack correspond to days with no warmer temperatures ahead
        # These are already set to 0 in the `answer` array
        
        return answer

```

## Next Greater Element I

**Solution**:

1. **Mapping `nums1` to Indices**:
   - Use a dictionary to map elements of `nums1` to their indices for efficient updates.

2. **Monotonic Decreasing Stack**:
   - Traverse `nums2` while maintaining a stack of indices.
   - The stack stores indices of elements in decreasing order of their values.
   - When a greater element is found:
     - Pop indices from the stack and update the result for the corresponding elements.

3. **Default Value**:
   - Initialize the result array with `-1` for cases where no greater element exists.

```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Initialize the answer array with -1 (default when no next greater element exists)
        ans = [-1] * len(nums1)

        # Create a mapping of elements in nums1 to their indices
        nums_12_map = {num: i for i, num in enumerate(nums1)}

        # Monotonic stack to find the next greater element
        stack = []

        # Traverse nums2 to find next greater elements
        for j in range(len(nums2)):
            # While the stack is not empty and the current number is greater than
            # the number corresponding to the index at the top of the stack
            while stack and nums2[j] > nums2[stack[-1]]:
                # Check if the number at the top of the stack exists in nums1
                if nums2[stack[-1]] in nums_12_map:
                    # Update the answer for the corresponding index in nums1
                    ans[nums_12_map[nums2[stack[-1]]]] = nums2[j]
                stack.pop()  # Pop the index from the stack

            # Push the current index onto the stack
            stack.append(j)

        # Return the answer array with next greater elements for nums1
        return ans
    
```

## Next Greater Element II

**Solution**:
1. **Simulating a Circular Array**:
   - Traverse the array twice (using indices `i % len(nums)`), allowing you to simulate the circular nature of the array.

2. **Monotonic Decreasing Stack**:
   - Use a stack to store indices of elements in decreasing order.
   - When a greater element is found:
     - Pop indices from the stack and update the result for those indices.

3. **Answer Array**:
   - Initialize the result array with `-1` for elements with no next greater value.

4. **Avoid Redundant Indices**:
   - Add indices to the stack only during the first pass (first \(n\) iterations).

```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        if len(nums) == 1:
            return [-1]  # A single element has no greater element

        # Initialize the answer array with -1 (default when no next greater element exists)
        ans = [-1] * len(nums)

        # Monotonic stack to store indices
        stack = []

        # Traverse the array twice (to simulate circular behavior)
        for i in range(2 * len(nums)):
            index = i % len(nums)  # Get the actual index in the circular array
            
            # While the stack is not empty and the current element is greater
            # than the element at the index stored at the top of the stack
            while stack and nums[index] > nums[stack[-1]]:
                pre_index = stack.pop()  # Pop the index of the smaller element
                ans[pre_index] = nums[index]  # Update the next greater element
            
            # Only add indices from the first pass
            if i < len(nums):
                stack.append(index)

        return ans

```

## Trapping Rain Water

**Solution**:

- The water trapped at each index depends on the **minimum of the left and right boundaries** surrounding it.
- Use a **monotonic decreasing stack** to efficiently identify the boundaries and calculate the water trapped.

---

## Algorithm

1. **Initialize**:
   - A `stack` to store indices of the bars.
   - A variable `volume` to accumulate the total trapped water.

2. **Traverse the Array**:
   - For each bar at index `i`:
     1. While the stack is not empty and the current bar's height is greater than the height of the bar at the top of the stack:
        - Pop the top of the stack as the "bottom" of the trapped water.
        - If the stack becomes empty, break (no left boundary exists).
        - Calculate the water trapped above this bottom:
          - Height of trapped water = `min(left, right) - height[bottom]`.
          - Width of trapped water = `i - stack[-1] - 1`.
        - Add this water to `volume`.

     2. Push the current index onto the stack.

3. **Return the Result**:
   - The total trapped water is stored in `volume`.


```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if len(height) < 3:  # Less than 3 bars cannot trap any water
            return 0
        
        stack = []  # Monotonic stack to store indices of the bars
        volume = 0  # Total water trapped

        # Traverse through the heights
        for i in range(len(height)):
            # While the current height is greater than the height of the bar at the top of the stack
            while stack and height[i] > height[stack[-1]]:
                bottom = stack.pop()  # The bar at the top of the stack serves as the bottom of the trapped water
                
                if not stack:
                    break  # No left boundary for the water

                # Calculate water trapped above the current bottom bar
                left = height[stack[-1]]  # Height of the left boundary
                right = height[i]  # Height of the right boundary
                h = min(left, right) - height[bottom]  # Effective height of trapped water
                w = i - stack[-1] - 1  # Width between the left and right boundaries
                volume += h * w  # Accumulate the water volume

            # If the current height is the same as the height at the stack's top, pop it (optional)
            if stack and height[i] == height[stack[-1]]:
                stack.pop()

            stack.append(i)  # Push the current index onto the stack

        return volume

```

## 

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # Initialize a stack to keep track of indices of heights
        stack = []
        max_area = 0
        
        # Append a zero-height bar to ensure all elements in the stack get processed
        # [4, 6, 8]
        heights.append(0)
        
        for i, h in enumerate(heights):
            # Ensure the stack maintains a non-decreasing order of heights
            while stack and heights[stack[-1]] > h:
                # Pop the top element (height index)
                height = heights[stack.pop()]
                
                # Calculate the width
                # If the stack is empty, the width is the current index i
                # This happens when there are no smaller heights to the left,
                # meaning the rectangle extends from index 0 to index i.
                width = i if not stack else i - stack[-1] - 1
                
                # Update the maximum area
                max_area = max(max_area, height * width)
            
            # Push the current index onto the stack
            stack.append(i)
        
        return max_area
    
```

# TOP 150

## Merge Sorted Array

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Merges two sorted arrays nums1 and nums2 into nums1 in-place.
        """
        # Pointers for nums1, nums2, and the position to insert in nums1
        p1 = m - 1  # Last valid element in nums1
        p2 = n - 1  # Last element in nums2
        p = m + n - 1  # Last position in nums1

        # Merge nums1 and nums2 from the back
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]  # Place nums1[p1] at the current position
                p1 -= 1
            else:
                nums1[p] = nums2[p2]  # Place nums2[p2] at the current position
                p2 -= 1
            p -= 1  # Move the insertion pointer backward

        # Copy remaining elements from nums2, if any
        while p2 >= 0:
            nums1[p] = nums2[p2]
            p2 -= 1
            p -= 1

```

## Remove Duplicates from Sorted Array

``` python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        # Initialize two pointers
        slow, fast = 0, 1

        # Traverse the array with the fast pointer
        while fast < len(nums):
            # If a new unique element is found
            if nums[slow] != nums[fast]:
                slow += 1  # Move the slow pointer
                nums[slow] = nums[fast]  # Copy the unique element to the slow pointer's position

            fast += 1  # Always increment the fast pointer

        # Return the length of the unique portion of the array
        return slow + 1
    
```

## Majority Element

### Boyer-Moore Voting Algorithm

**Key Idea**:

 Use a counter to track a potential majority element (`candidate`).
  - Traverse the array:
    - If the counter is `0`, set the current element as the `candidate`.
    - If the current element matches the `candidate`, increment the counter.
    - Otherwise, decrement the counter.

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        candidate = None
        count = 0

        # Phase 1: Find a candidate
        for num in nums:
            if count == 0:
                candidate = num
            count += 1 if num == candidate else -1

        # Phase 2: (Optional) Verify the candidate
        # If the problem guarantees that a majority element always exists, skip this step.
        # count = sum(1 for num in nums if num == candidate)
        # if count > len(nums) // 2:
        #     return candidate

        return candidate
``` 

## Remove Duplicates from Sorted Array II

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 1  # If there's only one element, return 1
        
        slow, fast = 0, 1  # Initialize two pointers
        count = 0  # Track the count of duplicates

        # Traverse the array using the fast pointer
        while fast < len(nums):
            if nums[slow] == nums[fast]:
                if count == 0:  # Allow the duplicate if it appears at most twice
                    count = 1
                    slow += 1
                    nums[slow] = nums[fast]
            else:  # If the elements are different
                count = 0  # Reset the count
                slow += 1
                nums[slow] = nums[fast]  # Update the position in the array
            
            fast += 1  # Move the fast pointer forward
        
        return slow + 1  # Return the new length of the array

```

**Optimization**:

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 2:
            return len(nums)  # Arrays of size 1 or 2 are already valid

        slow = 1  # Start from the second element

        # Traverse the array from the third element onward
        for fast in range(2, len(nums)):
            # If the current element is different from the element two steps back
            if nums[fast] != nums[slow - 1]:
                slow += 1
                nums[slow] = nums[fast]  # Move the current element to the valid position

        return slow + 1  # Return the new length of the modified array

```

## Rotate Array

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Rotates the array to the right by k steps.
        This modifies nums in-place.
        """
        def reverse(start: int, end: int) -> None:
            # Helper function to reverse elements in nums[start:end+1]
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1

        n = len(nums)
        k %= n  # Handle cases where k > n

        # Step 1: Reverse the entire array
        reverse(0, n - 1)

        # Step 2: Reverse the first k elements
        reverse(0, k - 1)

        # Step 3: Reverse the remaining n-k elements
        reverse(k, n - 1)
        
```



## Roman to Integer

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        # Mapping of Roman numerals to integers
        roman2int = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }

        # Edge case: Empty string
        if not s:
            return 0

        result = 0
        prev_value = 0

        # Iterate through the Roman numeral string
        for char in s:
            cur_value = roman2int[char]

            # If the current value is greater than the previous value, apply subtractive rule
            if cur_value > prev_value:
                result += cur_value - 2 * prev_value
            else:
                result += cur_value

            # Update previous value for the next iteration
            prev_value = cur_value

        return result

```

##

```python
def intToRoman(self, num: int) -> str:
    # Mapping of Roman numeral values to symbols in descending order
    value_to_symbol = [
        (1000, 'M'),
        (900, 'CM'),
        (500, 'D'),
        (400, 'CD'),
        (100, 'C'),
        (90, 'XC'),
        (50, 'L'),
        (40, 'XL'),
        (10, 'X'),
        (9, 'IX'),
        (5, 'V'),
        (4, 'IV'),
        (1, 'I')
    ]

    # StringBuilder to build the Roman numeral
    result = []

    # Iterate through the value-to-symbol mapping
    for value, symbol in value_to_symbol:
        # Determine the number of times the current value fits into num
        while num >= value:
            result.append(symbol)  # Append the corresponding Roman numeral symbol
            num -= value  # Reduce num by the value of the symbol

    return ''.join(result)
    
```

##

```python
def removeElement(self, nums: List[int], val: int) -> int:
    # Initialize a pointer for the next position to overwrite
    slow = 0

    # Traverse through the array
    for fast in range(len(nums)):
        # If the current element is not equal to val, keep it
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1

    # Return the new length of the array after removing val
    return slow

```

##  Insert Delete GetRandom O(1)

```python
class RandomizedSet:

    def __init__(self):
        """
        Initialize the data structure.
        - `values`: List to store the elements for random access.
        - `indices`: Dictionary to map element values to their indices for O(1) lookup and removal.
        """
        self.values = []  # List to hold elements
        self.indices = {}  # Dictionary to map values to their indices

    def insert(self, val: int) -> bool:
        """
        Inserts a value into the set. Returns True if the value was not already present, False otherwise.
        """
        if val in self.indices:
            return False  # Value already exists, insertion failed
        self.indices[val] = len(self.values)  # Store the index of the new value
        self.values.append(val)  # Add the value to the list
        return True

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns True if the value was present, False otherwise.
        """
        if val not in self.indices:
            return False  # Value not found, removal failed

        # Get the index of the element to remove
        index = self.indices[val]
        # Get the last element in the list
        last = self.values[-1]

        # Replace the element to remove with the last element
        self.values[index] = last
        # Update the index of the last element in the dictionary
        self.indices[last] = index

        # Remove the last element from the list
        self.values.pop()
        # Delete the removed element from the dictionary
        del self.indices[val]

        return True

    def getRandom(self) -> int:
        """
        Returns a random element from the set.
        """
        return random.choice(self.values)

```

## Valid Palindrome

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        # Helper function to check if a character is alphanumeric
        def isAlphaNum(char: str):
            # Check if the character is a digit ('0'-'9'), uppercase ('A'-'Z'), or lowercase ('a'-'z')
            return (ord('A') <= ord(char) <= ord('Z') or
                    ord('a') <= ord(char) <= ord('z') or
                    ord('0') <= ord(char) <= ord('9'))
        
        # Initialize two pointers: one at the start and the other at the end of the string
        start, end = 0, len(s) - 1

        # Iterate until the two pointers meet
        while start < end:
            # Move the `start` pointer forward until an alphanumeric character is found
            while start < end and not isAlphaNum(s[start]):
                start += 1

            # Move the `end` pointer backward until an alphanumeric character is found
            while start < end and not isAlphaNum(s[end]):
                end -= 1

            # Compare the characters at `start` and `end`, ignoring case
            if s[start].lower() != s[end].lower():
                return False  # Return False if characters don't match

            # Move both pointers inward
            start += 1
            end -= 1
        
        # If all characters matched, return True
        return True

```

## Minimum Size Subarray Sum

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        # Initialize pointers and variables
        left = 0
        cur_sum = 0  # Current sum of the window
        min_len = float('inf')  # Initialize min_len to infinity for comparison

        # Iterate through the array with the right pointer
        for right in range(len(nums)):
            cur_sum += nums[right]  # Add the current number to the window sum

            # Shrink the window from the left as long as the condition is met
            while cur_sum >= target:
                # Update the minimum length
                min_len = min(min_len, right - left + 1)
                cur_sum -= nums[left]  # Remove the leftmost element
                left += 1  # Move the left pointer forward

        # Return the minimum length if found; otherwise, return 0
        return min_len if min_len != float('inf') else 0

```

## Longest Substring Without Repeating Characters

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
    # Initialize variables
    char_set = set()  # To store unique characters in the current window
    left = 0  # Left pointer for the sliding window
    max_length = 0  # To track the length of the longest substring

    # Iterate through the string with the right pointer
    for right in range(len(s)):
        # Shrink the window if the current character is already in the set
        while s[right] in char_set:
            char_set.remove(s[left])  # Remove the leftmost character
            left += 1  # Move the left pointer forward

        # Add the current character to the set
        char_set.add(s[right])

        # Update the maximum length of the substring
        max_length = max(max_length, right - left + 1)

    return max_length

```

## Product Except Self

**Solution**:
prefix and suffix product

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # Initialize the result array with all elements as 1
        ans = [1] * len(nums)

        # First pass: Compute the prefix product for each index
        # ans[i] will store the product of all elements to the left of `i`
        for i in range(1, len(ans)):
            ans[i] = ans[i-1] * nums[i-1]  # Multiply the previous prefix with nums[i-1]

        # Initialize a suffix product variable
        suffix = 1
        # Second pass: Compute the suffix product for each index
        # Multiply the suffix product with the corresponding prefix product
        for i in range(len(ans)-2, -1, -1):  # Start from the second-to-last element
            suffix *= nums[i+1]  # Update the suffix product
            ans[i] *= suffix     # Multiply the prefix product stored in `ans[i]` with the suffix

        return ans

```

## Longest Common Prefix

p.s., s[:count] will return the entire string when count > len(s)

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:  # Handle edge case for empty input
            return ""

        pref = strs[0]  # Start with the first string as the prefix
        count = len(pref)  # Track the length of the prefix

        for s in strs[1:]:  # Compare the prefix with each subsequent string
            while pref[:count] != s[:count]:
                count -= 1  # Reduce the length of the prefix
                if count == 0:  # If prefix becomes empty, no common prefix exists
                    return ""
        
        return pref[:count]

```

## Longest Consecutive Sequence

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:  # Handle edge case for empty array
            return 0
        
        num_set = set(nums)
        longest_streak = 0

        for num in num_set:
            # Only start counting if `num` is the beginning of a sequence
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1

                # Count consecutive numbers
                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1

                # Update the longest streak
                longest_streak = max(longest_streak, current_streak)

        return longest_streak

```

## Rotate Image

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Rotate the input NxN matrix 90 degrees clockwise in-place.
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)  # Size of the matrix
        top, bottom, left, right = 0, n-1, 0, n-1  # Initialize boundaries

        # Loop to process layers from the outermost to the innermost
        while n > 1:
            # Rotate the current layer
            for i in range(n-1):  # Process the elements in the current layer
                # Save the top-left element temporarily
                element = matrix[top][left + i]
                
                # Perform 4-way rotation:
                # 1. Move element from left column to top row
                matrix[top][left + i] = matrix[bottom - i][left]
                # 2. Move element from bottom row to left column
                matrix[bottom - i][left] = matrix[bottom][right - i]
                # 3. Move element from right column to bottom row
                matrix[bottom][right - i] = matrix[top + i][right]
                # 4. Move saved element to right column
                matrix[top + i][right] = element

            # Move to the next inner layer by updating boundaries
            n -= 2
            top += 1
            bottom -= 1
            left += 1
            right -= 1

```

## Set Matrix Zeroes

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # Get the number of rows and columns in the matrix
        row, col = len(matrix), len(matrix[0])

        # Initialize flags for rows and columns to track where zeros exist
        row_flag = [0] * row
        col_flag = [0] * col

        # First pass: Identify rows and columns that need to be zeroed
        for i in range(row):
            for j in range(col):
                # If an element is zero, mark its row and column in the flags
                if matrix[i][j] == 0:
                    row_flag[i] = 1
                    col_flag[j] = 1

        # Second pass: Update the matrix based on the flags
        for i in range(row):
            for j in range(col):
                # If the current row or column is marked, set the element to zero
                if row_flag[i] or col_flag[j]:
                    matrix[i][j] = 0

```

## Summary Ranges

```python
class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        result = []  # List to store the resulting ranges

        slow = 0  # Initialize the slow pointer to traverse the array
        while slow < len(nums):  # Continue until all elements are processed
            fast = slow  # Start the fast pointer at the same position as slow

            # Expand the fast pointer as long as consecutive numbers are found
            while fast + 1 < len(nums) and nums[fast + 1] == nums[fast] + 1:
                fast += 1

            # If there is a range (more than one element), add it in the format "start->end"
            if fast > slow:
                result.append(str(nums[slow]) + '->' + str(nums[fast]))
            else:
                # If there's only one element, add it as a single number
                result.append(str(nums[slow]))

            # Move the slow pointer to the next unprocessed element
            fast += 1
            slow = fast

        return result

```

## Game of Life

```python
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Updates the board in-place to the next state of the Game of Life.
        """
        # Get the number of rows and columns
        row, col = len(board), len(board[0])

        # Define the 8 possible directions of neighbors
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # Top-left, Top, Top-right
            (0, -1),         (0, 1),     # Left, Right
            (1, -1), (1, 0), (1, 1)      # Bottom-left, Bottom, Bottom-right
        ]

        # First pass: Calculate the next state and encode it in the current board
        for i in range(row):
            for j in range(col):
                liveNeighbors = 0

                # Count live neighbors
                for x, y in directions:
                    if 0 <= i + x < row and 0 <= j + y < col:  # Check boundaries
                        # Use % 10 to only consider the current state
                        if (board[i + x][j + y]) % 10 == 1:
                            liveNeighbors += 1

                # Encode the next state by adding (liveNeighbors * 10)
                # The next state is stored in the tens place
                board[i][j] += (liveNeighbors * 10)

        # Second pass: Decode the next state and update the board
        for i in range(row):
            for j in range(col):
                currentState = board[i][j] % 10  # Extract the current state
                nextState = board[i][j] // 10    # Extract the next state (encoded in tens place)

                if currentState == 0:  # Dead cell
                    # Dead cell becomes live if it has exactly 3 live neighbors
                    if nextState == 3:
                        board[i][j] = 1
                    else:
                        board[i][j] = 0
                else:  # Live cell
                    # Live cell dies if it has fewer than 2 or more than 3 live neighbors
                    if nextState < 2 or nextState > 3:
                        board[i][j] = 0
                    else:  # Live cell survives if it has 2 or 3 live neighbors
                        board[i][j] = 1

```

## Zigzag Conversion

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        # If there's only one row or the number of rows is greater than or equal
        # to the length of the string, no zigzag is needed. Return the string as is.
        if numRows == 1 or numRows >= len(s):
            return s

        # Initialize a list of lists to hold characters for each row
        result = [[] for _ in range(numRows)]

        # Start from the first row, and set the initial direction to 'down'
        row, direction = 0, 1

        # Iterate through each character in the string
        for char in s:
            # Append the character to the current row
            result[row].append(char)

            # Change direction at the top or bottom row
            if row == 0:
                direction = 1  # Move down
            elif row == numRows - 1:
                direction = -1  # Move up
            
            # Move to the next row in the current direction
            row += direction

        # Flatten the 2D list and join characters to form the final result string
        return ('').join([char for line in result for char in line])

```

## Contains Duplicate II

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        # Dictionary to store the last seen index of each number
        num_dict = {}

        for i, num in enumerate(nums):
            # Check if the number exists in the dictionary and the index difference is within k
            if num in num_dict and i - num_dict[num] <= k:
                return True
            
            # Update the dictionary with the current index of the number
            num_dict[num] = i

        # No nearby duplicates found
        return False

```

## Insert Interval

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        if not intervals:
            return [newInterval]  # Edge case: empty list

        result = []

        for i, interval in enumerate(intervals):
            # If the current interval is completely before the newInterval
            if interval[1] < newInterval[0]:
                result.append(interval)
            # If the current interval is completely after the newInterval
            elif interval[0] > newInterval[1]:
                # Add the newInterval and remaining intervals
                return result + [newInterval] + intervals[i:]
            # If intervals overlap, merge them into newInterval
            else:
                newInterval[0] = min(interval[0], newInterval[0])
                newInterval[1] = max(interval[1], newInterval[1])

        # Add the final merged newInterval
        return result + [newInterval]

```

## Container With Most Water

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        # Initialize two pointers at the ends of the list
        left, right = 0, len(height) - 1

        # Variable to keep track of the maximum area found
        max_area = 0

        # Iterate until the two pointers meet
        while left < right:
            # Calculate the current area using the shorter height
            # and the distance between the two pointers
            area = min(height[left], height[right]) * (right - left)

            # Update the maximum area if the current area is larger
            max_area = max(max_area, area)

            # Move the pointer pointing to the shorter height inward
            # This is because the limiting factor for the area is the shorter height
            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1

        # Return the maximum area found
        return max_area

```

## Group Anagrams

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # If the input list is empty, return an empty list
        if not strs:
            return []

        # Dictionary to group words by their sorted character string
        group = {}

        # Iterate over each word in the input list
        for word in strs:
            # Sort the characters of the word to create a key
            # Anagrams will have the same sorted key
            sorted_word = ''.join(sorted(word))

            # If the sorted key is not in the dictionary, add it with the current word as the first value
            if sorted_word not in group:
                group[sorted_word] = [word]
            else:
                # If the key exists, append the current word to the list
                group[sorted_word].append(word)

        # Collect all grouped anagrams into a result list
        result = []
        for key, value in group.items():
            # Append each group (list of anagrams) to the result
            result.append(value)

        # Return the final grouped anagrams
        return result


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        if not strs:
            return []  # Return an empty list if input is empty
        
        group = {}  # Dictionary to group anagrams

        for word in strs:
            # Initialize character frequency count for the word
            count = [0] * 26  # For 26 lowercase English letters
            for char in word:
                count[ord(char) - ord('a')] += 1

            # Convert the frequency count into a hashable key
            key = tuple(count)  # Use a tuple instead of string for better efficiency

            # Group the words by their character frequency key
            if key not in group:
                group[key] = [word]
            else:
                group[key].append(word)

        # Return the grouped anagrams
        return list(group.values())

```

## Add Two Numbers

```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        # Initialize a dummy node to build the result list
        # `dummy` serves as a placeholder for the start of the result linked list
        dummy = ListNode()
        result = dummy  # Keep a reference to the head of the result list

        total, carry = 0, 0  # Initialize total and carry to 0

        # Iterate as long as there are nodes in l1, l2, or a carry to process
        while l1 or l2 or carry:
            total = carry  # Start with the carry from the previous digit

            # Add the value of the current node in l1, if it exists
            if l1:
                total += l1.val
                l1 = l1.next  # Move to the next node in l1

            # Add the value of the current node in l2, if it exists
            if l2:
                total += l2.val
                l2 = l2.next  # Move to the next node in l2

            # Calculate the value for the current digit and update the carry
            val = total % 10  # The current digit
            carry = total // 10  # Carry-over for the next digit

            # Append the current digit to the result linked list
            dummy.next = ListNode(val)
            dummy = dummy.next  # Move the pointer to the newly created node

        # Return the next node of the dummy, which is the head of the result list
        return result.next

```

## Simplify Path

```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        # Initialize a stack to simulate directory traversal
        stack = []

        # Pointer to iterate through the path string
        current = 0

        # Process the path character by character
        while current < len(path):
            # Skip consecutive slashes
            if path[current] == '/':
                current += 1
            else:
                # Identify the start of a directory/file name
                start = current
                # Find the end of the directory/file name (until the next '/')
                while current < len(path) and path[current] != '/':
                    current += 1
                # Extract the directory/file name
                stack.append(path[start:current])

                # Process the extracted element
                element = stack.pop()
                if element == '.':
                    # Ignore '.' as it refers to the current directory
                    continue
                elif element == '..':
                    # '..' means go up one level; pop from the stack if it's not empty
                    if stack:
                        stack.pop()
                else:
                    # A valid directory/file name; push it onto the stack
                    stack.append(element)

        # Reconstruct the simplified canonical path
        return '/' + ('/').join(stack)

```

## Merge Two Sorted Lists

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # Create a dummy node to serve as the starting point for the merged list
        dummy = ListNode()
        result = dummy  # Keep a reference to the head of the merged list

        # Traverse both lists until one is exhausted
        while list1 and list2:
            if list1.val < list2.val:
                # If list1's current node has a smaller value, append it to the merged list
                dummy.next = list1
                list1 = list1.next  # Move to the next node in list1
            else:
                # Otherwise, append list2's current node to the merged list
                dummy.next = list2
                list2 = list2.next  # Move to the next node in list2
            dummy = dummy.next  # Move to the next position in the merged list

        # Append any remaining nodes from list1 or list2
        if list1:
            dummy.next = list1
        if list2:
            dummy.next = list2

        # Return the merged list, skipping the dummy node
        return result.next

```

## Copy List with Random Pointer

```python
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None
        
        # Step 1: Create a mapping of original nodes to their copies
        node_map = {}
        cur = head
        
        # First pass: Copy all nodes (without setting `random` yet)
        while cur:
            node_map[cur] = Node(cur.val)
            cur = cur.next
        
        # Step 2: Set the `next` and `random` pointers for the copied nodes
        cur = head
        while cur:
            if cur.next:
                node_map[cur].next = node_map[cur.next]
            if cur.random:
                node_map[cur].random = node_map[cur.random]
            cur = cur.next
        
        # Step 3: Return the copied head node
        return node_map[head]


class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None  # If the list is empty, return None
        
        # Step 1: Interleave copied nodes
        # For each node in the original list, create a new node with the same value
        # Insert the new node immediately after the original node
        cur = head
        while cur:
            # Create a new node (deep copy) with the same value as the current node
            new_node = Node(cur.val, cur.next)
            # Insert the new node right after the current node
            cur.next = new_node
            # Move to the next original node
            cur = new_node.next
        
        # Step 2: Assign random pointers
        # For each copied node, set its random pointer based on the original node's random pointer
        cur = head
        while cur:
            if cur.random:
                # The copied node's random pointer should point to the copy of the original random node
                cur.next.random = cur.random.next
            # Move to the next original node (skipping the copied node)
            cur = cur.next.next
        
        # Step 3: Separate the copied list from the original
        # Restore the original list and extract the copied list
        cur = head
        copied_head = head.next  # The head of the copied list
        while cur:
            # Get the copied node
            copied = cur.next
            # Restore the original list by skipping the copied node
            cur.next = copied.next
            # Link the copied node to the next copied node
            if copied.next:
                copied.next = copied.next.next
            # Move to the next original node
            cur = cur.next
        
        return copied_head  # Return the head of the copied list

```

## Valid Sudoku

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # Create hash sets for rows, columns, and subgrids
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        subgrids = [set() for _ in range(9)]  # Each subgrid indexed by (row // 3) * 3 + (col // 3)

        for row in range(9):
            for col in range(9):
                num = board[row][col]
                if num == '.':
                    continue  # Skip empty cells
                
                # Calculate the index of the subgrid
                subgrid_index = (row // 3) * 3 + (col // 3)

                # Check for duplicates
                if (
                    num in rows[row] or
                    num in cols[col] or
                    num in subgrids[subgrid_index]
                ):
                    return False

                # Add the number to the corresponding row, column, and subgrid
                rows[row].add(num)
                cols[col].add(num)
                subgrids[subgrid_index].add(num)
        
        return True

```

## Min Stack

```python
class MinStack:
    def __init__(self):
        self.stack = []  # Stack to hold tuples of (value, current_min)

    def push(self, val: int) -> None:
        # If stack is empty, the current min is the value itself
        current_min = val if not self.stack else min(val, self.stack[-1][1])
        self.stack.append((val, current_min))  # Push value and updated min

    def pop(self) -> None:
        self.stack.pop()  # Remove the top element

    def top(self) -> int:
        return self.stack[-1][0]  # Return the top value of the stack

    def getMin(self) -> int:
        return self.stack[-1][1]  # Return the current minimum value

```


## Pow(x, n)

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        # Handle base cases: x^0 = 1 and 0^n = 0 for n > 0
        if x == 0:  # Any number 0 raised to a positive power is 0
            return 0
        if n == 0:  # Any number raised to the power of 0 is 1
            return 1

        # Define a helper function for recursive calculation of power
        def halfPow(x: float, n: int) -> float:
            if n == 0:  # Base case: power of 0 is 1
                return 1
            # Recursively calculate the result for n // 2
            result = halfPow(x, n // 2)
            # Combine results based on whether n is even or odd
            return result * result if n % 2 == 0 else result * result * x

        # Compute power for the absolute value of n
        ans = halfPow(x, abs(n))
        
        # If n is negative, take the reciprocal of the result
        return ans if n >= 0 else 1 / ans

```

## Remove Duplicates from Sorted List II

```python
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # Base case: If the list is empty, return None
        if not head:
            return None

        # Base case: If the list has only one node, return the head
        if head and not head.next:
            return head

        # Create a dummy node that points to the head of the list
        # This helps handle edge cases where the first few nodes are duplicates
        dummy = ListNode(next=head)

        # Initialize two pointers:
        # 'slow' tracks the last non-duplicate node
        # 'fast' scans ahead to identify duplicates
        slow, fast = dummy, head.next

        # Traverse the list until the 'fast' pointer reaches the end
        while fast:
            # Case 1: Current nodes are not duplicates
            if fast and slow.next.val != fast.val:
                slow = slow.next  # Move 'slow' pointer forward
                fast = fast.next  # Move 'fast' pointer forward
            else:
                # Case 2: Current nodes are duplicates
                # Skip all nodes with the same value as 'slow.next'
                while fast and slow.next.val == fast.val:
                    fast = fast.next
                
                # Remove the duplicates by pointing 'slow.next' to the node after the duplicates
                slow.next = fast

                # Move the 'fast' pointer forward if it's not None
                fast = fast.next if fast else fast

        # Return the new head of the list, which is the node after the dummy node
        return dummy.next

```

## Rotate List

```python
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or not head.next or k == 0:
            return head

        # Step 1: Find the length of the list
        length = 1
        cur = head
        while cur.next:
            cur = cur.next
            length += 1

        # Step 2: Normalize k
        k = k % length
        if k == 0:
            return head  # No rotation needed

        # Step 3: Find the new tail (n - k - 1) and new head (n - k)
        cur = head
        for _ in range(length - k - 1):
            cur = cur.next

        # Step 4: Rearrange pointers
        new_head = cur.next
        cur.next = None  # Break the list
        tail = new_head
        while tail and tail.next:
            tail = tail.next
        tail.next = head  # Connect the tail to the original head

        return new_head

```

## Word Pattern

```python
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        words = s.split()
        # Length mismatch check
        if len(pattern) != len(words):
            return False
        # Ensure one-to-one mapping using set comparison
        return len(set(zip(pattern, words))) == len(set(pattern)) == len(set(words))

```

## Sort List

```python
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # Helper function to merge two sorted linked lists
        def merge(left: Optional[ListNode], right: Optional[ListNode]) -> Optional[ListNode]:
            dummy = ListNode()  # Dummy node to simplify the merge logic
            cur = dummy

            # Compare nodes from left and right lists, adding the smaller one to the merged list
            while left and right:
                if left.val < right.val:
                    cur.next = left
                    left = left.next
                else:
                    cur.next = right
                    right = right.next
                cur = cur.next

            # Add any remaining nodes from either list
            if left:
                cur.next = left
            if right:
                cur.next = right

            return dummy.next

        # Helper function to find the middle of the linked list and split it into two halves
        def getMid(head: Optional[ListNode]) -> Optional[ListNode]:
            prev, slow, fast = None, head, head

            # Use the slow and fast pointer approach to find the middle
            while fast and fast.next:
                prev = slow
                slow = slow.next
                fast = fast.next.next

            # Disconnect the first half from the second half
            prev.next = None

            return slow

        # Base case: if the list is empty or has a single node, it's already sorted
        if not head or not head.next:
            return head

        # Split the list into two halves
        mid = getMid(head)

        # Recursively sort both halves
        left = self.sortList(head)
        right = self.sortList(mid)

        # Merge the two sorted halves
        return merge(left, right)

```

## Partition List

```python
class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        if not head or not head.next:
            return head

        dummy = ListNode(next=head)  # Dummy node for managing the original list
        prev, cur = dummy, head

        new = ListNode()  # New list to hold nodes >= x
        newCur = new

        # Traverse the list and partition nodes
        while cur:
            if cur.val >= x:
                # Remove the current node from the original list and add it to the new list
                prev.next = cur.next
                newCur.next = cur
                cur = cur.next
                newCur = newCur.next
                newCur.next = None  # Disconnect the new node from the original list
            else:
                prev = cur
                cur = cur.next

        # Connect the end of the original list to the new list
        prev.next = new.next

        return dummy.next

```

## Factorial Trailing Zeroes

```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        # Initialize the result variable to count the number of trailing zeros
        res = 0

        # Loop to count factors of 5 in the numbers from 1 to n
        while n != 0:
            # Add the number of multiples of 5 to the result
            res += n // 5

            # Update n by dividing it by 5 to count higher powers of 5 (e.g., 25, 125)
            n //= 5

        # Return the total count of trailing zeros
        return res

```

## Same Tree

```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
         # If both nodes are None, the trees are the same
        if not p and not q:
            return True

        # If one node is None and the other isn't, the trees are not the same
        if not p or not q or p.val != q.val:
            return False

        # Recursively check if left subtrees and right subtrees are the same
        left = self.isSameTree(p.left, q.left)
        right = self.isSameTree(p.right, q.right)
        
        # Trees are the same only if both left and right subtrees are identical
        return left and right 
        
```

## Reverse Linked List II

```python
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        # If the list has only one node or no reversal is needed, return the head
        if not head.next or left == right:
            return head

        # Create a dummy node to simplify edge cases (e.g., reversing from the head)
        dummy = ListNode(0, next=head)
        traversal = dummy

        # Move traversal pointer to the node just before the 'left' position
        for _ in range(left - 1):
            traversal = traversal.next

        # Start is the first node to reverse
        start = traversal.next

        # Reverse the sublist between 'left' and 'right'
        prev, cur = None, start
        for _ in range(right - left + 1):
            nxt = cur.next  # Temporarily store the next node
            cur.next = prev  # Reverse the pointer
            prev = cur  # Move prev to the current node
            cur = nxt  # Move cur to the next node

        # Connect the reversed sublist back to the original list
        traversal.next = prev  # Connect the node before 'left' to the new head of the reversed sublist
        start.next = cur  # Connect the tail of the reversed sublist to the node after 'right'

        # Return the updated list starting from the dummy node's next pointer
        return dummy.next

```

## Find Peak Element

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        # Binary search to find a peak
        left, right = 0, len(nums) - 1

        while left < right:
            mid = (left + right) // 2

            # If mid is less than its right neighbor, move to the right half
            if nums[mid] < nums[mid + 1]:
                left = mid + 1
            else:
                # Otherwise, move to the left half (including mid)
                right = mid

        # At the end of the loop, left == right, pointing to a peak element
        return left

```

## LRU Cache

```python
class ListNode:
    def __init__(self, key: int = 0, value: int = 0, prev_node: Optional['ListNode'] = None, next_node: Optional['ListNode'] = None):
        # Node to represent a doubly linked list element with key-value pair
        self.key = key
        self.value = value
        self.prev = prev_node
        self.next = next_node

class LRUCache:
    def __init__(self, capacity: int):
        # Initialize the cache with a fixed capacity
        self.capacity = capacity
        self.size = 0
        self.cache = {}  # Dictionary to map keys to nodes for O(1) access
        
        # Create dummy head and tail nodes for the doubly linked list
        self.head = ListNode()
        self.tail = ListNode()

        # Connect head and tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Optional[ListNode]) -> None:
        """Remove a node from the doubly linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_tail(self, node: Optional[ListNode]) -> None:
        """Add a node to the tail of the doubly linked list (most recently used)."""
        self.tail.prev.next = node
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev = node

    def get(self, key: int) -> int:
        """
        Get the value of the key if it exists in the cache.
        Move the accessed node to the tail as it is the most recently used.
        """
        if key not in self.cache:
            return -1  # Key not found in the cache

        # Access the node
        node = self.cache[key]

        # Move the node to the tail
        self._remove_node(node)
        self._add_tail(node)

        return node.value

    def put(self, key: int, value: int) -> None:
        """
        Add a key-value pair to the cache.
        If the key already exists, update its value and move it to the tail.
        If the cache exceeds capacity, remove the least recently used node.
        """
        if key in self.cache:
            # Key exists: Update the value and move node to the tail
            node = self.cache[key]
            node.value = value

            self._remove_node(node)
            self._add_tail(node)
        else:
            # Key does not exist: Check capacity and add new node
            if self.size == self.capacity:
                # Remove the least recently used node (head.next)
                lru_node = self.head.next
                self.cache.pop(lru_node.key)
                self._remove_node(lru_node)
                self.size -= 1

            # Create a new node and add it to the tail
            node = ListNode(key, value)
            self._add_tail(node)
            self.cache[key] = node
            self.size += 1

```

## Construct Quad Tree

```python
"""
# Definition for a QuadTree node.
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
"""

class Solution:
    def construct(self, grid: List[List[int]]) -> 'Node':
        """
        Constructs a QuadTree from a 2D grid of integers.
        
        :param grid: 2D list of integers (0 or 1) representing the input grid
        :return: Root node of the QuadTree
        """

        def isSame(left: int, right: int, up: int, bottom: int) -> bool:
            """
            Checks if all elements in a subgrid are the same.
            
            :param left: Left column index of the subgrid
            :param right: Right column index of the subgrid
            :param up: Upper row index of the subgrid
            :param bottom: Lower row index of the subgrid
            :return: True if all elements are the same, False otherwise
            """
            firstElement = grid[up][left]  # Reference element to compare against

            # Iterate through the subgrid
            for row in grid[up:bottom+1]:
                for element in row[left:right+1]:
                    if firstElement != element:  # If any element differs, return False
                        return False
            return True

        def constructQuad(left: int, right: int, up: int, bottom: int) -> 'Node':
            """
            Recursively constructs the QuadTree for a given subgrid.
            
            :param left: Left column index of the subgrid
            :param right: Right column index of the subgrid
            :param up: Upper row index of the subgrid
            :param bottom: Lower row index of the subgrid
            :return: Root node of the constructed QuadTree for the subgrid
            """
            node = Node()  # Create a new QuadTree node
            node.val = grid[up][left]  # Set the value of the node to the top-left element of the subgrid

            # If all elements in the subgrid are the same, make this node a leaf
            if isSame(left, right, up, bottom):
                node.isLeaf = True
                return node

            # Otherwise, split the grid into four quadrants and recursively construct each
            node.isLeaf = False
            mid_col = (left + right) // 2  # Midpoint for splitting columns
            mid_row = (up + bottom) // 2  # Midpoint for splitting rows
            
            # Top-left quadrant
            node.topLeft = constructQuad(left, mid_col, up, mid_row)
            # Top-right quadrant
            node.topRight = constructQuad(mid_col + 1, right, up, mid_row)
            # Bottom-left quadrant
            node.bottomLeft = constructQuad(left, mid_col, mid_row + 1, bottom)
            # Bottom-right quadrant
            node.bottomRight = constructQuad(mid_col + 1, right, mid_row + 1, bottom)
            
            return node  # Return the constructed node

        # Start constructing the QuadTree for the entire grid
        return constructQuad(0, len(grid[0]) - 1, 0, len(grid) - 1)

```

## Add Binary

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        carry = 0  # Initialize carry to 0
        result = []  # List to store the result in reverse order

        # Start from the last digit of both strings
        m, n = len(a) - 1, len(b) - 1
        
        while m >= 0 or n >= 0 or carry:  # Process until all digits and carry are handled
            # Extract current digits or use 0 if out of range
            num1 = int(a[m]) if m >= 0 else 0
            num2 = int(b[n]) if n >= 0 else 0
            
            # Calculate the total sum and carry
            total = carry + num1 + num2
            carry = total // 2  # Update carry for the next iteration
            digit = total % 2  # Extract the binary digit to add to the result
            
            # Append the digit to the result
            result.append(str(digit))
            
            # Move to the next left digits
            m -= 1
            n -= 1
        
        # Reverse the result since the binary digits were appended in reverse order
        return ''.join(result[::-1])  # Join and reverse the result list

```

## Reverse Bits

```python
class Solution:
    def reverseBits(self, n: int) -> int:
        result = 0  # To store the reversed bits
        
        # Iterate over all 32 bits
        for i in range(32):
            # Extract the last bit of n
            last_bit = n & 1
            
            # Shift result to the left to make space for the new bit
            result = (result << 1) | last_bit
            
            # Right shift n to process the next bit
            n >>= 1
        
        return result

```

e.g., Bitwise AND (&); Bitwise OR (|); Bitwise XOR (^); Bitwise NOT (~); Left Shift (<<); Right Shit (>>)

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0  # Initialize a counter to count the number of 1 bits
        
        # Iterate through all bits of the number
        while n != 0:
            # Extract the last (rightmost) bit using bitwise AND with 1
            bit = n & 1
            
            # If the extracted bit is 1, increment the counter
            if bit == 1:
                count += 1
            
            # Right shift n by 1 to process the next bit
            n >>= 1
        
        return count  # Return the total count of 1 bits

```

## Sum Root to Leaf Numbers

```python
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def traversal(cur: Optional[TreeNode], num: int) -> None:
            # Base case: If it's a leaf node, add the formed number to the result
            if not cur.left and not cur.right:
                result.append(num + cur.val)  # Add the current value to form the number
                return

            # If there is a left child, recurse on the left subtree
            if cur.left:
                # Append current node's value and shift to the left
                traversal(cur.left, (num + cur.val) * 10)

            # If there is a right child, recurse on the right subtree
            if cur.right:
                # Append current node's value and shift to the right
                traversal(cur.right, (num + cur.val) * 10)

        # Initialize an empty list to store all root-to-leaf numbers
        result = []
        traversal(root, 0)  # Start the traversal from the root with an initial value of 0
        return sum(result)  # Return the sum of all the numbers


class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def traversal(cur: Optional[TreeNode], num: int) -> int:
            if not cur:
                return 0

            num = num * 10 + cur.val

            # If it's a leaf node, return the current number
            if not cur.left and not cur.right:
                return num

            # Recur for left and right subtrees and accumulate the sum
            return traversal(cur.left, num) + traversal(cur.right, num)

        return traversal(root, 0)

```

## Flatten Binary Tree to Linked List

```python
class Solution:
    def __init__(self):
        # This variable stores the previously visited node
        self.prev = None

    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Flatten the binary tree to a linked list in-place (right child represents the next node).
        
        The traversal order is **post-order**, starting from the right subtree, then the left subtree.
        """
        if not root:
            return  # Base case: If the node is None, do nothing

        # Recursively flatten the right subtree first
        self.flatten(root.right)

        # Recursively flatten the left subtree
        self.flatten(root.left)

        # Reorganize the current node:
        # 1. The current node's right pointer points to the previous flattened node (self.prev).
        # 2. The current node's left pointer is set to None (linked list only uses right pointers).
        root.right = self.prev
        root.left = None

        # Update the previous node to the current node
        self.prev = root

```

## Single Number

```python
"""
a ^ a = 0
a ^ 0 = a
"""

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        """
        Find the single number in the list where every other number appears twice.
        """
        # Initialize the result with the first element
        unique = nums[0]

        # XOR all remaining elements in the list
        for num in nums[1:]:
            unique ^= num  # XOR operation: cancel out duplicate numbers

        return unique  # Return the single non-duplicate number

```

## Binary Search Tree Iterator

```python
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        # Initialize the iterator with the root of the BST
        self.cur = root  # Pointer to the current node
        self.stack = []  # Stack to keep track of nodes for in-order traversal
        

    def next(self) -> int:
        # Traverse to the leftmost node
        while self.cur:
            self.stack.append(self.cur)  # Push the current node onto the stack
            self.cur = self.cur.left

        # Pop the node from the stack (the next smallest element)
        self.cur = self.stack.pop()
        value = self.cur.val  # Store the value to return

        # Move to the right subtree for the next call
        self.cur = self.cur.right

        return value


    def hasNext(self) -> bool:
        # There is a next element if either the stack is not empty
        # or the current pointer is not null
        return self.cur is not None or len(self.stack) > 0

```

## Binary Tree Zigzag Level Order Traversal

```python
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []
        flip = False  # Track the direction of traversal
        traversal = deque([root])  # Queue for BFS

        while traversal:
            size = len(traversal)
            level = deque()  # Use deque to construct level directly in the correct order

            for _ in range(size):
                node = traversal.popleft()

                # Add node values in the correct order based on `flip`
                if flip:
                    level.appendleft(node.val)
                else:
                    level.append(node.val)

                # Add child nodes to the queue
                if node.left:
                    traversal.append(node.left)
                if node.right:
                    traversal.append(node.right)

            result.append(list(level))  # Convert deque to list for the result
            flip = not flip  # Toggle the direction

        return result
        
```

## Find Minimum in Rotated Sorted Array

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # Initialize pointers to the start and end of the array
        left, right = 0, len(nums) - 1

        # Perform binary search
        while left < right:
            # Calculate the middle index
            mid = (left + right) // 2

            # Compare the middle element with the rightmost element
            if nums[mid] > nums[right]:
                # If nums[mid] is greater than nums[right],
                # the minimum is in the right half of the array
                left = mid + 1
            else:
                # If nums[mid] is less than or equal to nums[right],
                # the minimum is in the left half (inclusive of mid)
                right = mid

        # When the loop exits, 'left' will point to the minimum element
        return nums[left]

```

## Kth Smallest Element in a BST

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        # Initialize the current node pointer and a stack for iterative traversal
        current = root
        stack = []

        # Perform in-order traversal
        while current or stack:
            # Traverse to the leftmost node
            while current:
                stack.append(current)
                current = current.left

            # Process the node at the top of the stack
            current = stack.pop()
            k -= 1  # Decrement k since we've visited one more node
            
            # If k becomes 0, we've found the k-th smallest element
            if k == 0:
                return current.val

            # Move to the right subtree
            current = current.right

        # If the loop ends without finding the k-th smallest element, return -1
        # This can happen if k is invalid (e.g., larger than the number of nodes)
        return -1

```

## Kth Largest Element in an Array

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # Initialize an empty min-heap
        min_heap = []

        # Iterate through each number in the array
        for num in nums:
            # Push the current number onto the min-heap
            heapq.heappush(min_heap, num)

            # If the size of the heap exceeds k, remove the smallest element
            # This ensures that the heap only contains the k largest elements
            if len(min_heap) > k:
                heapq.heappop(min_heap)

        # The root of the heap (min_heap[0]) is the k-th largest element
        return min_heap[0]

```

## Generate Parentheses

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def backtracking(path: List[str], left: int, right: int) -> None:
            # If the current combination is valid and complete, add it to the result
            if left == right == n:
                result.append(''.join(path))  # Join the path list to form the final string
                return

            # Add a '(' if we haven't used all left parentheses
            if left < n:
                path.append('(')
                backtracking(path, left + 1, right)
                path.pop()  # Backtrack: remove the last added '('

            # Add a ')' if it won't invalidate the sequence
            if right < left:
                path.append(')')
                backtracking(path, left, right + 1)
                path.pop()  # Backtrack: remove the last added ')'

        # Initialize result and start backtracking
        result = []
        backtracking([], 0, 0)
        return result

```

## Snakes and Ladders

```python
class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        n = len(board)  # Size of the board (n x n)

        # Dictionary to store the minimum steps required to reach each square
        step_dict = {1: 0}
        queue = deque([1])  # BFS queue initialized with the starting square

        while queue:
            cur = queue.popleft()

            # If we reach the last square, return the number of steps
            if cur == n * n:
                return step_dict[cur]

            # Explore all possible moves (dice rolls from 1 to 6)
            for then in range(cur + 1, min(cur + 6, n * n) + 1):
                # Convert 1D board position to 2D indices
                row, col = self.get(then, n)

                # If the square has a ladder or snake, update the destination
                if board[row][col] != -1:
                    then = board[row][col]

                # If the square hasn't been visited, add it to the BFS queue
                if then not in step_dict:
                    step_dict[then] = step_dict[cur] + 1
                    queue.append(then)

        # If the queue is exhausted and we didn't reach the last square, return -1
        return -1

    def get(self, step: int, n: int) -> Tuple[int, int]:
        """
        Convert a 1-based board position to 2D indices (row, col).
        Handles the alternating direction of rows in the board.
        """
        row = (n - 1) - (step - 1) // n  # Calculate the row (bottom to top)
        
        # Determine the column based on the direction of the row
        if row % 2 == n % 2:
            col = (n - 1) - (step - 1) % n
        else:
            col = (step - 1) % n

        return row, col

```

## Minimum Genetic Mutation

```python
class Solution:
    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        # Convert bank to a set for O(1) lookups
        bank = set(bank)

        # If the end gene is not in the bank, return -1
        if endGene not in bank:
            return -1

        # BFS initialization
        queue = deque([(startGene, 0)])  # (current gene, mutation steps)
        visited = set([startGene])      # Set of visited genes

        while queue:
            gene, step = queue.popleft()

            # If we reach the end gene, return the number of steps
            if gene == endGene:
                return step

            # Generate all possible mutations
            for i in range(len(gene)):  # Iterate through all positions
                for mutation in "ACGT":
                    # Skip if the mutation is the same as the current character
                    if mutation == gene[i]:
                        continue
                    
                    # Create the new gene by mutating the i-th character
                    new_gene = gene[:i] + mutation + gene[i + 1:]

                    # If the new gene is valid and not visited, add to the queue
                    if new_gene not in visited and new_gene in bank:
                        queue.append((new_gene, step + 1))
                        visited.add(new_gene)  # Mark as visited

        # If no path to the end gene was found, return -1
        return -1

```

## Find K Pairs with Smallest Sums

```python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        # Handle edge cases where input arrays are empty or k is zero
        if not nums1 or not nums2 or k == 0:
            return []

        # Min-heap to store pairs with their sums, initialized as empty
        heap = []
        result = []  # List to store the resulting k pairs

        # Initialize the heap with the smallest elements from nums1 paired with the first element of nums2
        # Only take up to the first k elements from nums1 to ensure efficiency
        for i in range(min(k, len(nums1))):
            # Push tuples of the form (sum, index in nums1, index in nums2) into the heap
            heapq.heappush(heap, (nums1[i] + nums2[0], i, 0))

        # Extract the k smallest pairs from the heap
        while heap and len(result) < k:
            # Pop the smallest sum pair from the heap
            curr_sum, i, j = heapq.heappop(heap)
            # Add the corresponding pair (nums1[i], nums2[j]) to the result
            result.append([nums1[i], nums2[j]])

            # If there's a next element in nums2 for the current nums1[i], push it into the heap
            if j + 1 < len(nums2):
                # Push the new pair (sum, i, j+1) into the heap
                heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))

        # Return the result containing the k smallest pairs
        return result

```

## Single Number II

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # Initialize an array to count the bits at each of the 32 positions
        bits = [0] * 32
       
        # Iterate through each number in the input list
        for num in nums:
            for i in range(32):
                # Extract the i-th bit of the number and add it to the corresponding position in 'bits'
                bits[i] += (num >> i) & 1

        # Variable to store the result
        ans = 0
        for i in range(32):
            # Reconstruct the number by taking modulo 3 of each bit position
            # If bits[i] % 3 is 1, this bit belongs to the single number
            ans |= (bits[i] % 3) << i

        # Handle negative numbers (convert from unsigned 32-bit representation to signed integer)
        # If the 31st bit (sign bit) is set, the number is negative
        return ans if ans < 2 ** 31 else ans - 2 ** 32

```

## Number of Islands

```python
class Solution:
    """
    DFS
    """
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(row: int, col: int) -> None:
            # Check boundary conditions
            if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]):
                return
            
            # Stop if the cell is water ("0")
            if grid[row][col] == "0":
                return

            # Mark the cell as visited
            grid[row][col] = "0"

            # Explore all four directions
            for x, y in directions:
                dfs(row + x, col + y)

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Directions: up, down, left, right
        num = 0  # Number of islands
        m, n = len(grid), len(grid[0])  # Grid dimensions

        for row in range(m):
            for col in range(n):
                if grid[row][col] == "1":  # Found a new island
                    num += 1  # Increment the island count
                    dfs(row, col)  # Start DFS to mark the entire island

        return num


class Solution:
    """
    BFS
    """
    def numIslands(self, grid: List[List[str]]) -> int:
        num = 0  # Count of islands
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Possible movement directions
        m, n = len(grid), len(grid[0])  # Dimensions of the grid

        # Iterate through all cells in the grid
        for row in range(m):
            for col in range(n):
                # If the current cell is land, start BFS to explore the island
                if grid[row][col] == "1":
                    num += 1  # Increment the island count
                    queue = deque([(row, col)])  # Initialize the BFS queue
                    
                    # Perform BFS to visit all connected land cells
                    while queue:
                        x, y = queue.popleft()
                        grid[x][y] = "0"  # Mark the cell as visited
                        
                        # Explore all four directions
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy  # Calculate neighbor coordinates
                            
                            # Check bounds and whether the neighbor is land
                            if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == "1":
                                queue.append((nx, ny))  # Add neighbor to the queue
                                grid[nx][ny] = "0"  # Mark the neighbor as visited

        return num  # Return the total number of islands

```

## Surrounded Regions

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        def bfs(row: int, col: int):
            # Initialize a queue for BFS and mark the starting cell as visited
            queue = deque([(row, col)])
            board[row][col] = 'T'  # Temporarily mark the cell to avoid revisiting

            while queue:
                r, c = queue.popleft()
               
                # Explore all four possible directions (up, down, left, right)
                for x, y in directions:
                    if 0 <= r + x < m and 0 <= c + y < n and board[r + x][c + y] == 'O':
                        queue.append((r + x, c + y))  # Add the neighbor to the queue
                        board[r + x][c + y] = 'T'  # Mark as visited

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Define movement directions
        m, n = len(board), len(board[0])  # Dimensions of the board

        # Step 1: Start BFS from all border 'O' cells and mark them as 'T'
        for row in range(m):
            for col in range(n):
                # Only start BFS for 'O' cells on the border
                if (row == 0 or row == m - 1 or col == 0 or col == n - 1) and board[row][col] == 'O':
                    bfs(row, col)

        # Step 2: Convert all remaining 'O' cells to 'X' (these are the captured regions)
        for row in range(m):
            for col in range(n):
                if board[row][col] == 'O':
                    board[row][col] = 'X'

        # Step 3: Convert all 'T' cells back to 'O' (these are the border-connected regions)
        for row in range(m):
            for col in range(n):
                if board[row][col] == 'T':
                    board[row][col] = 'O'

```

## Trie

```python
class Trie:

    def __init__(self):
        # Initialize the root of the Trie as an empty dictionary
        self.root = {}
        

    def insert(self, word: str) -> None:
        """
        Insert a word into the Trie.
        Each character in the word is added as a key in nested dictionaries.
        The end of the word is marked with a special key '#'.
        """
        cur = self.root
        for char in word:
            # Create a new dictionary for the character if it doesn't exist
            if char not in cur:
                cur[char] = {}
            cur = cur[char]  # Move to the next level
        cur['#'] = True  # Mark the end of the word
        

    def search(self, word: str) -> bool:
        """
        Search for a word in the Trie.
        Returns True if the word exists and is marked as complete with '#'.
        """
        cur = self.find(word)  # Use the helper method to navigate to the word
        return True if cur and '#' in cur else False
        

    def startsWith(self, prefix: str) -> bool:
        """
        Check if any word in the Trie starts with the given prefix.
        Returns True if the prefix exists in the Trie.
        """
        return True if self.find(prefix) else False 


    def find(self, prefix: str) -> dict:
        """
        Helper function to navigate through the Trie for a given prefix.
        Returns the last node corresponding to the prefix if it exists, otherwise None.
        """
        cur = self.root
        for char in prefix:
            # If the character is not in the current level, return None
            if char not in cur:
                return None
            cur = cur[char]  # Move to the next level
        return cur  # Return the final node corresponding to the prefix

```

## Clone Graph

```python
"""
BFS
"""
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        # If the input graph is empty, return None
        if not node:
            return None

        # A mapping to store the original node to its cloned counterpart
        mapping = {}
        
        # Use a queue for Breadth-First Search (BFS) traversal
        queue = deque([node])
        
        # Create the clone for the input node and add it to the mapping
        mapping[node] = Node(node.val)

        # Perform BFS
        while queue:
            # Get the next node from the queue
            current = queue.popleft()

            # Traverse all neighbors of the current node
            for neighbor in current.neighbors:
                # If the neighbor has not been cloned yet
                if neighbor not in mapping:
                    # Clone the neighbor and add it to the mapping
                    queue.append(neighbor)
                    mapping[neighbor] = Node(neighbor.val)
                
                # Append the cloned neighbor to the current node's clone's neighbors
                mapping[current].neighbors.append(mapping[neighbor])

        # Return the clone of the starting node
        return mapping[node]


"""
DFS
"""

class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        # Recursive function to perform DFS and clone the graph
        def dfs(node: Optional['Node']) -> Optional['Node']:
            # Base case: If the node is None, return None
            if not node:
                return None

            # If the node has already been cloned, return the clone
            if node in mapping:
                return mapping[node]

            # Create a clone of the current node
            clone = Node(node.val)
            # Store the clone in the mapping to avoid duplication
            mapping[node] = clone

            # Recursively clone all the neighbors
            for neighbor in node.neighbors:
                clone.neighbors.append(dfs(neighbor))

            # Return the cloned node
            return clone

        # Dictionary to map original nodes to their cloned counterparts
        mapping = {}
        
        # Start the cloning process from the input node
        return dfs(node)

```

## Reverse Nodes in k-Group

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # If the list is empty, return None
        if not head:
            return None

        # Dummy node to simplify edge case handling (e.g., reversing the first group)
        dummy = ListNode(val=0, next=head)

        # Slow and fast pointers for group traversal
        slow = fast = dummy
        while fast:
            # Move the fast pointer ahead by k+1 nodes to ensure there are enough nodes to reverse
            fast = slow
            for _ in range(k + 1):
                if not fast:  # If there are fewer than k nodes, return the processed list
                    return dummy.next
                fast = fast.next

            # Reverse the k nodes between slow and fast pointers
            prev, cur = None, slow.next
            start = slow.next  # Start node of the current group
            while cur != fast:  # Reverse until reaching the fast pointer
                then = cur.next  # Store the next node
                cur.next = prev  # Reverse the current node's pointer
                prev = cur  # Move prev forward
                cur = then  # Move cur forward

            # Connect the reversed group to the rest of the list
            slow.next = prev  # Connect the previous group to the start of the reversed group
            start.next = cur  # Connect the end of the reversed group to the next group
            slow = start  # Move slow to the start node (which is now the end of the reversed group)

        return dummy.next  # Return the new head of the list

```

## Evaluate Division

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        def dfs(num1: str, num2: str, visited: set) -> float:
            # If the starting variable is not in the graph, return -1.0
            if num1 not in graph:
                return -1.0

            # Base case: If num1 and num2 are the same, return 1.0 (self-division)
            if num1 == num2:
                return 1.0

            visited.add(num1)  # Mark the current node as visited

            # Explore neighbors
            for neighbor, value in graph[num1].items():
                if neighbor in visited:  # Skip already visited nodes
                    continue
                if neighbor == num2:  # Direct connection to the target
                    return value
                # Recursive DFS call
                ans = dfs(neighbor, num2, visited)
                if ans != -1.0:  # If a valid path is found
                    return value * ans

            return -1.0  # No valid path found

        # Step 1: Build the graph
        graph = {}
        for i, (numerator, denominator) in enumerate(equations):
            if numerator not in graph:
                graph[numerator] = {}
            if denominator not in graph:
                graph[denominator] = {}
            graph[numerator][denominator] = values[i]
            graph[denominator][numerator] = 1.0 / values[i]

        # Step 2: Process queries
        result = []
        for num1, num2 in queries:
            # Check if both variables are in the graph
            if num1 not in graph or num2 not in graph:
                result.append(-1.0)  # If either variable is missing, result is -1.0
            else:
                visited = set()  # Fresh visited set for each query
                result.append(dfs(num1, num2, visited))

        return result

```

## N-Queens II

```python
class Solution:
    def __init__(self):
        # Initialize the result to count the number of valid N-Queens solutions
        self.result = 0

    def totalNQueens(self, n: int) -> int:
        def check(board: List[List[int]], row: int, col: int) -> bool:
            """
            Check if placing a queen at (row, col) is valid.
            A position is valid if there are no other queens:
            - In the same column
            - On the upper-left diagonal
            - On the upper-right diagonal
            """
            # Check the same column in rows above
            for i in range(row - 1, -1, -1):
                if board[i][col] == 1:
                    return False
            
            # Check the upper-left diagonal
            for i, j in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
                if board[i][j] == 1:
                    return False

            # Check the upper-right diagonal
            for i, j in zip(range(row - 1, -1, -1), range(col + 1, n)):
                if board[i][j] == 1:
                    return False
            
            # If all checks pass, the position is valid
            return True

        def backtracing(board: List[List[int]], row: int):
            """
            Try to place queens row by row using backtracking.
            - If all rows are filled (row == n), count it as a valid solution.
            - Otherwise, try placing a queen in each column of the current row
              and recursively solve the problem for the next row.
            """
            if row == n:  # Base case: all rows are filled
                self.result += 1  # Found a valid configuration
                return

            # Try placing a queen in each column of the current row
            for col in range(n):
                if check(board, row, col):  # Check if placing at (row, col) is valid
                    board[row][col] = 1  # Place the queen
                    backtracing(board, row + 1)  # Move to the next row
                    board[row][col] = 0  # Backtrack: remove the queen

        # Initialize the board as an n x n grid filled with zeros (no queens placed)
        board = [[0] * n for _ in range(n)]
        
        # Start the backtracking process from the first row
        backtracing(board, 0)

        # Return the total number of valid solutions
        return self.result


class Solution:
    def totalNQueens(self, n: int) -> int:
        """
        Key Ideas:
        - A queen can attack another if they share the same row, column, or diagonal.
        - To track conflicts:
          - Use `cols` to store columns where queens are placed.
          - Use `diagonals` to store major diagonals, identified by `row - col`.
          - Use `anti_diagonals` to store minor diagonals, identified by `row + col`.
        """

        def backtrack(row: int):
            if row == n:  # All queens are placed
                self.result += 1
                return
            
            for col in range(n):
                # Skip invalid positions based on column and diagonal constraints
                if col in cols or (row - col) in diagonals or (row + col) in anti_diagonals:
                    continue
                
                # Place the queen
                cols.add(col)
                diagonals.add(row - col)
                anti_diagonals.add(row + col)
                
                # Recurse to the next row
                backtrack(row + 1)
                
                # Remove the queen (backtrack)
                cols.remove(col)
                diagonals.remove(row - col)
                anti_diagonals.remove(row + col)
        
        # Initialize tracking sets
        cols = set()  # Tracks columns with queens
        diagonals = set()  # Tracks major diagonals (row - col)
        anti_diagonals = set()  # Tracks minor diagonals (row + col)
        
        # Initialize result counter
        self.result = 0
        
        # Start backtracking from the first row
        backtrack(0)
        
        return self.result


```

## Triangle

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # Start from the second-to-last row and move upwards
        for row in range(len(triangle) - 2, -1, -1):
            for col in range(len(triangle[row])):
                # Update the current cell with the minimum path sum
                triangle[row][col] += min(triangle[row + 1][col], triangle[row + 1][col + 1])

        # The top element now contains the minimum path sum
        return triangle[0][0]


class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # Initialize dp with the last row of the triangle
        dp = triangle[-1][:]

        # Process rows from the second-to-last to the top
        for row in range(len(triangle) - 2, -1, -1):
            for col in range(len(triangle[row])):
                # Update dp[col] with the minimum path sum for the current position
                dp[col] = triangle[row][col] + min(dp[col], dp[col + 1])

        # The result is stored at the top of dp
        return dp[0]
        
```

## Substring with Concatenation of All Words

```python
from typing import List

class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        # Helper function to compare two dictionaries for equality
        def isSame(dict1: dict, dict2: dict) -> bool:
            for key, value in dict2.items():
                if key not in dict1:  # Key missing in dict1
                    return False
                if value != dict1[key]:  # Value mismatch
                    return False
            return True

        result = []  # List to store starting indices of valid substrings
        word_len = len(words[0])  # Length of each word in the words list
        word_count = len(words)  # Total number of words
        substring_size = word_len * word_count  # Total size of the concatenated substring

        # Create a frequency map of the words in the list
        word_map = {}
        for word in words:
            word_map[word] = word_map.get(word, 0) + 1

        # Iterate through all possible starting positions modulo word_len
        for i in range(word_len):
            current_map = {}  # Current frequency map of words in the window
            start, end = i, i  # Sliding window pointers
            match_count = 0  # Count of matched words in the current window

            # Slide the window across the string
            while end + word_len <= len(s):
                candidate = s[end:end + word_len]  # Extract a word of length word_len
                end += word_len  # Move the end pointer

                if candidate in word_map:
                    # If the word is valid, add it to the current map
                    current_map[candidate] = current_map.get(candidate, 0) + 1
                    match_count += 1

                    # If the number of matched words equals the number of words in the list
                    if match_count == word_count:
                        # Check if the frequency maps match
                        if isSame(current_map, word_map):
                            result.append(start)  # Add the start index to the result
                        
                        # Adjust the window to remove the leftmost word
                        start_word = s[start:start + word_len]
                        current_map[start_word] -= 1
                        match_count -= 1
                        start = start + word_len
                else:
                    # If the candidate word is not valid, reset the window
                    current_map.clear()
                    match_count = 0
                    start = end  # Move the start pointer to the end

        return result

```

## Course Schedule

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        def dfs(node: int) -> bool:
            # If the node is currently being visited, a cycle is detected
            if node in visit and visit[node] == 0:
                return False

            # If the node has already been fully processed, skip it
            if node in visit and visit[node] == 1:
                return True

            # Mark the node as being visited
            visit[node] = 0
            for neighbor in graph[node]:
                # Recursively visit all neighbors; if any detects a cycle, return False
                if not dfs(neighbor):
                    return False

            # Mark the node as fully processed
            visit[node] = 1
            return True

        # Build the graph as an adjacency list
        graph = {i: [] for i in range(numCourses)}
        for course, prerequisite in prerequisites:
            graph[prerequisite].append(course)

        # Dictionary to track visit state:
        # 0 = visiting, 1 = fully processed, not in `visit` = unvisited
        visit = {}

        # Perform DFS for all nodes in the graph
        for key in graph:
            if not dfs(key):
                return False  # If any DFS call detects a cycle, return False

        # If no cycles are found, all courses can be completed
        return True

```

## Course Schedule II

```python
from collections import defaultdict, deque

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # DFS function
        def dfs(course: int) -> bool:
            if visit[course] == VISITING:  # Cycle detected
                return False
            if visit[course] == PROCESSED:  # Already processed
                return True

            # Mark as visiting
            visit[course] = VISITING
            for neighbor in graph[course]:
                if not dfs(neighbor):
                    return False

            # Mark as processed and add to result
            visit[course] = PROCESSED
            result.appendleft(course)  # Append to the left for topological order
            return True

        # Edge case: no prerequisites
        if not prerequisites:
            return list(range(numCourses))
        
        # Constants for visit states
        UNVISITED, VISITING, PROCESSED = 0, 1, 2
        
        # Build the graph
        graph = defaultdict(list)
        for course, prerequisite in prerequisites:
            graph[prerequisite].append(course)

        # Initialize visit states and result
        visit = [UNVISITED] * numCourses
        result = deque()  # Use deque for efficient prepending

        # Perform DFS for all courses
        for course in range(numCourses):
            if visit[course] == UNVISITED:
                if not dfs(course):
                    return []  # Return empty if a cycle is detected

        return list(result)

```

## Minimum Window Substring

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # If the source string is shorter than the target string, no valid window can exist
        if len(s) < len(t):
            return ""

        # Create a frequency map for characters in t
        need_map = Counter(t)
        # Number of unique characters in t that need to be fully matched in the window
        required = len(need_map)

        # Initialize pointers for the sliding window
        left = 0
        # Tracks how many unique characters in t are currently fully matched in the window
        formed = 0
        # Dictionary to track characters in the current window
        window_map = {}

        # Variable to store the smallest window substring
        min_window = ""
        
        # Iterate through the source string with the right pointer
        for right in range(len(s)):
            char = s[right]
            # Add the current character to the window map
            window_map[char] = window_map.get(char, 0) + 1

            # If the character matches the required count in need_map, increment `formed`
            if char in need_map and window_map[char] == need_map[char]:
                formed += 1

            # Try to contract the window from the left while it remains valid
            while left <= right and formed == required:
                char = s[left]

                # Update the minimum window if this window is smaller
                if min_window == "":
                    min_window = s[left:right+1]
                else:
                    min_window = s[left:right+1] if right - left + 1 < len(min_window) else min_window

                # Remove the leftmost character from the window
                window_map[char] -= 1
                # If removing the character breaks the validity of the window, decrement `formed`
                if char in need_map and window_map[char] < need_map[char]:
                    formed -= 1

                # Move the left pointer forward to shrink the window
                left += 1

        # Return the smallest window found, or an empty string if no valid window exists
        return min_window

```

## Minimum Path Sum

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        # Use a single-row DP array to save space
        dp = [0] * n
        dp[0] = grid[0][0]  # Initialize the first cell
        
        # Fill the first row
        for col in range(1, n):
            dp[col] = dp[col - 1] + grid[0][col]
        
        # Process the remaining rows
        for row in range(1, m):
            dp[0] += grid[row][0]  # Update the first column
            for col in range(1, n):
                dp[col] = min(dp[col], dp[col - 1]) + grid[row][col]
        
        return dp[-1]

```

## Search in Rotated Sorted Array

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # Helper function for binary search
        def binarySearch(arr: List[int], target: int, offset: int = 0) -> int:
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid + offset
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1

        # Edge case: single element array
        if len(nums) == 1:
            return 0 if nums[0] == target else -1

        # Find the pivot where the rotation occurs
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid

        # Left now points to the pivot
        pivot = left

        # Perform binary search in both subarrays divided by the pivot
        left_result = binarySearch(nums[:pivot], target, offset=0)
        right_result = binarySearch(nums[pivot:], target, offset=pivot)

        # Return the result
        if left_result != -1:
            return left_result
        if right_result != -1:
            return right_result
        return -1


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2

            # If the target is found, return its index
            if nums[mid] == target:
                return mid

            # Determine which half is sorted
            if nums[left] <= nums[mid]:  # Left half is sorted
                if nums[left] <= target < nums[mid]:  # Target is in the left half
                    right = mid - 1
                else:  # Target is in the right half
                    left = mid + 1
            else:  # Right half is sorted
                if nums[mid] < target <= nums[right]:  # Target is in the right half
                    left = mid + 1
                else:  # Target is in the left half
                    right = mid - 1

        # If we exit the loop, the target is not in the array
        return -1


```

## Interleaving String

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # If the lengths of s1 and s2 combined do not match the length of s3,
        # it is impossible for s3 to be an interleaving of s1 and s2.
        if len(s1) + len(s2) != len(s3):
            return False
        
        # Create a DP table where dp[i][j] indicates whether the first i characters
        # of s1 and the first j characters of s2 can form the first i+j characters of s3.
        dp = [[False] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        
        # Base case: Empty s1 and s2 form an empty s3.
        dp[0][0] = True
        
        # Fill the DP table.
        for i in range(len(s1) + 1):  # Iterate over all characters of s1 (including 0)
            for j in range(len(s2) + 1):  # Iterate over all characters of s2 (including 0)
                # Check if the current character of s1 matches the current character of s3
                # and if the previous state dp[i-1][j] is True.
                if i > 0 and s1[i - 1] == s3[i + j - 1]:
                    dp[i][j] |= dp[i - 1][j]  # Carry forward the result from dp[i-1][j]
                
                # Check if the current character of s2 matches the current character of s3
                # and if the previous state dp[i][j-1] is True.
                if j > 0 and s2[j - 1] == s3[i + j - 1]:
                    dp[i][j] |= dp[i][j - 1]  # Carry forward the result from dp[i][j-1]
        
        # The value at dp[len(s1)][len(s2)] tells whether s3 can be formed by interleaving
        # all characters of s1 and s2.
        return dp[len(s1)][len(s2)]

```

## Binary Tree Maximum Path Sum

```python
class Solution:
    def __init__(self):
        self.result = float('-inf')  # Initialize the result to negative infinity

    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        def traversal(node: Optional[TreeNode]) -> int:
            if not node:
                return 0  # Base case: a null node contributes 0 to the path sum

            # Recursively calculate the max path sums for left and right subtrees
            left_max = max(traversal(node.left), 0)  # Ignore negative contributions
            right_max = max(traversal(node.right), 0)  # Ignore negative contributions

            # Update the global result: maximum sum through the current node
            self.result = max(self.result, left_max + right_max + node.val)

            # Return the maximum path sum starting from the current node
            return max(left_max, right_max) + node.val

        traversal(root)  # Start traversal from the root node
        return self.result  # Return the maximum path sum

```

```python
"""
Two Pointers
"""

class Solution:
    def longestPalindrome(self, s: str) -> str:
        # Helper function to expand around the center and find the longest palindrome
        def countAroundCenter(left: int, right: int) -> str:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            # Return the palindrome substring found by expanding
            return s[left + 1:right]

        # Base case: If the string is empty or has one character, return it
        if len(s) <= 1:
            return s

        result = ""
        for i in range(len(s)):
            # Check for the longest odd-length palindrome centered at `i`
            odd_palindrome = countAroundCenter(i, i)
            # Check for the longest even-length palindrome centered at `i` and `i+1`
            even_palindrome = countAroundCenter(i, i + 1)

            # Choose the longer palindrome of the two
            current_longest = odd_palindrome if len(odd_palindrome) > len(even_palindrome) else even_palindrome

            # Update the result if the current longest is longer than the result
            result = current_longest if len(current_longest) > len(result) else result

        return result

```

## Maximal Square

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        # Edge case: if the matrix is empty or has no columns, return 0
        if not matrix or not matrix[0]:
            return 0

        # Dimensions of the matrix
        m, n = len(matrix), len(matrix[0])
        # DP table to store the size of the largest square ending at each cell
        dp = [[0] * n for _ in range(m)]
        # Variable to track the side length of the largest square found
        max_side = 0

        # Fill the DP table
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':  # Only process cells with '1'
                    if i == 0 or j == 0:  # Base case: first row or first column
                        dp[i][j] = 1  # Square size is 1 for these cells
                    else:
                        # Transition relation:
                        # The size of the square ending at (i, j) depends on the
                        # minimum of the squares ending at (i-1, j), (i, j-1), and (i-1, j-1)
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

                    # Update the maximum side length found so far
                    max_side = max(max_side, dp[i][j])

        # Optional: Print the DP table for debugging
        # print(dp)

        # The area of the largest square is the square of its side length
        return max_side ** 2

```

```python
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        # Helper function to check if two points are identical
        def isSame(a: List[int], b: List[int]) -> bool:
            return a[0] == b[0] and a[1] == b[1]

        # Helper function to calculate slope between two points
        def getSlope(a: List[int], b: List[int]) -> float:
            deltaX = a[0] - b[0]
            # Handle vertical lines (infinite slope)
            if deltaX == 0:
                return float('inf')

            deltaY = a[1] - b[1]
            return float(deltaY) / float(deltaX)

        # Base case: if 2 or fewer points, they're always on the same line
        if len(points) <= 2:
            return len(points)

        # Initialize result to 1 (a single point is always on a line)
        max_num = 1

        # For each point, find lines through it
        for i in range(len(points)):
            # Count duplicate points
            duplicate = 0
            # Dictionary to store count of points for each slope
            point_map = {}
            
            # Compare with all points after current point
            for j in range(i+1, len(points)):
                if isSame(points[i], points[j]):
                    # If points are identical, increment duplicate counter
                    duplicate += 1
                else:
                    # Calculate slope with current point
                    slope = getSlope(points[i], points[j])
                    # Initialize count to 1 (to include the endpoint) and increment
                    point_map[slope] = point_map.get(slope, 1) + 1

            # Current max is either all duplicates, or max points along a slope + duplicates
            current_max = duplicate
            if point_map:
                current_max = max(point_map.values()) + duplicate

            # Update global maximum
            max_num = max(num, current_max)

        return max_num

```

## Median of Two Sorted Arrays

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # Ensure nums1 is the smaller array to minimize binary search complexity
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1

        m, n = len(nums1), len(nums2)  # Lengths of the two arrays
        l = m + n  # Total length of the combined arrays
        half = (m + n + 1) // 2  # Half length for partitioning

        left, right = 0, m  # Binary search bounds on the smaller array

        while left <= right:
            partition1 = (left + right) // 2  # Partition index for nums1
            partition2 = half - partition1  # Partition index for nums2

            # Values around the partition for nums1
            left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
            right1 = float('inf') if partition1 == m else nums1[partition1]

            # Values around the partition for nums2
            left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
            right2 = float('inf') if partition2 == n else nums2[partition2]

            # Check if the current partition is valid
            if left1 <= right2 and left2 <= right1:
                # If the total length is odd, return the maximum of left values
                if l % 2 == 1:
                    return max(left1, left2)
                else:
                    # If the total length is even, return the average of max left and min right values
                    return (max(left1, left2) + min(right1, right2)) / 2

            # Adjust the binary search bounds
            if left1 > right2:
                right = partition1 - 1  # Move left in nums1
            if right1 < left2:
                left = partition1 + 1  # Move right in nums1

```

## Find Median from Data Stream

```python
import heapq

class MedianFinder:
    def __init__(self):
        # Max-heap for the smaller half
        self.small = []
        # Min-heap for the larger half
        self.large = []


    def addNum(self, num: int) -> None:
        # Add the number to the max-heap
        heapq.heappush(self.small, -num)

        # Ensure the max-heap's largest element is less than or equal to the min-heap's smallest element
        if self.small and self.large and -(self.small[0]) > self.large[0]:
            element = heapq.heappop(self.small)
            heapq.heappush(self.large, -element)

        # Ensure the heaps are balanced in size approximately (size difference at most 1)
        if len(self.small) > len(self.large) + 1:
            element = heapq.heappop(self.small)
            heapq.heappush(self.large, -element)
        elif len(self.large) > len(self.small) + 1:
            lement = heapq.heappop(self.large)
            heapq.heappush(self.large, -element)

    def findMedian(self) -> float:
        # If total number of elements is odd, return the root of the larger heap
        if len(self.small) > len(self.large):
            return -(self.small[0])
        elif len(self.large) > len(self.small):
            return self.large[0]
        else:
            # If total number of elements is even, return the average of the roots
            return (-(self.small[0]) + self.large[0]) / 2

```

## Word Labber

```python
from collections import deque
from typing import List

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # Convert the wordList into a set for O(1) lookups
        word_set = set(wordList)
        
        # If the endWord is not in the word set, there's no possible transformation
        if endWord not in word_set:
            return 0

        # Initialize a queue for BFS with the starting word and step count (1)
        queue = deque([(beginWord, 1)])
        
        # Perform BFS to find the shortest path
        while queue:
            # Dequeue the current word and its transformation step count
            word, step = queue.popleft()
            
            # If the current word matches the endWord, return the step count
            if word == endWord:
                return step

            # Generate all possible words by changing one character at a time
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    # Create a new word by replacing the character at position i
                    next_word = word[:i] + c + word[i+1:]

                    # If the new word is in the word set (valid transformation)
                    if next_word in word_set:
                        # Add the new word to the queue with an incremented step count
                        queue.append((next_word, step + 1))
                        
                        # Remove the new word from the word set to mark it as visited
                        word_set.remove(next_word)

        # If the queue is exhausted without finding the endWord, return 0
        return 0

```

## Merge k Sorted Lists

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists or all(node is None for node in lists):
            return None

        # Min-heap to store (value, index, node) tuples
        heap = []
        
        # Push the initial nodes of each list into the heap
        for i, node in enumerate(lists):
            if node:  # Only push non-empty lists
                heappush(heap, (node.val, i, node))
        
        dummy = ListNode()
        cur = dummy
        
        while heap:
            # Pop the smallest element from the heap
            _, i, min_node = heappop(heap)
            
            # Add the smallest node to the merged list
            cur.next = min_node
            cur = cur.next
            
            # If there's a next node in the same list, push it into the heap
            if min_node.next:
                heappush(heap, (min_node.next.val, i, min_node.next))
        
        return dummy.next

```

## IPO

```python
import heapq
from typing import List

class Solution:
    def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        # Step 1: Combine capital and profits into a list of tuples (capital, profit)
        # and sort by the capital required for each project in ascending order.
        projects = sorted(zip(capital, profits), key=lambda x: x[0])
        
        # Step 2: Max-heap to store the profits of projects that can be started.
        # Use negative profits to simulate a max-heap with Python's min-heap.
        maxProfit = []
        
        # Step 3: Iterate up to `k` times to pick at most `k` projects.
        n = len(projects)  # Total number of projects
        index = 0          # Tracks the current position in the sorted project list
        
        for _ in range(k):
            # Step 4: Add all projects to the heap whose capital requirement
            # is less than or equal to the current available capital `w`.
            while index < n and projects[index][0] <= w:
                # Push the profit (negated) into the max-heap
                heapq.heappush(maxProfit, -projects[index][1])
                index += 1

            # Step 5: If no projects can be started, break the loop.
            if not maxProfit:
                break
                
            # Step 6: Select the most profitable project from the heap.
            # Add its profit to the current capital `w`.
            w += -(heapq.heappop(maxProfit))
        
        # Step 7: Return the final capital after selecting up to `k` projects.
        return w

```

## Maximum Sum Circular Subarray

```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        # Edge case: Single element array
        if len(nums) == 1:
            return nums[0]

        # Initialize variables
        curMax, curMin = 0, 0
        globMax, globMin = nums[0], nums[0]
        total = 0  # Sum of all elements in the array

        # Iterate through the array
        for num in nums:
            total += num  # Update total sum

            # Update current maximum subarray sum
            curMax = max(curMax + num, num)
            globMax = max(globMax, curMax)  # Update global maximum

            # Update current minimum subarray sum
            curMin = min(curMin + num, num)
            globMin = min(globMin, curMin)  # Update global minimum

        # If the global maximum is positive, consider both circular and non-circular cases
        # Otherwise, return the global maximum (handles all-negative arrays)
        return max(globMax, total - globMin) if globMax > 0 else globMax

```

## Word Search

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        # Helper function to perform DFS from a specific cell
        def dfs(row: int, col: int, index: int) -> bool:
            # Base case: If the current index matches the last character of the word, return True
            if index == len(word) - 1:
                return True

            # Store the current character and mark the cell as visited
            char = board[row][col]
            board[row][col] = '#'  # Mark as visited by replacing it with a placeholder

            # Explore all 4 possible directions
            for x, y in directions:
                # Check if the neighboring cell is within bounds and matches the next character
                if 0 <= row + x < m and 0 <= col + y < n:
                    if board[row + x][col + y] == word[index + 1] and dfs(row + x, col + y, index + 1):
                        return True

            # Restore the original character after backtracking
            board[row][col] = char
            return False

        # Define the possible moves: down, up, left, right
        directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]

        # Get dimensions of the board
        m, n = len(board), len(board[0])

        # Iterate through each cell in the board to find the starting point
        for i in range(m):
            for j in range(n):
                # If the current cell matches the first character, start a DFS from it
                if board[i][j] == word[0] and dfs(i, j, 0):
                    return True

        # If no valid path is found, return False
        return False

```

## Design Add and Search Words Data Structure

```python
class WordDictionary:

    def __init__(self):
        # Initialize the root of the Trie as an empty dictionary
        self.root = {}
        

    def addWord(self, word: str) -> None:
        """
        Add a word to the Trie.
        """
        cur = self.root  # Start from the root node
        for char in word:
            # If the character is not in the current node, add it
            if char not in cur:
                cur[char] = {}
            # Move to the next node
            cur = cur[char]
        # Mark the end of a word with a special '#' key
        cur['#'] = True

        
    def search(self, word: str) -> bool:
        """
        Search for a word in the Trie. Supports wildcard '.' which matches any character.
        """
        def dfs(cur: dict, index: int) -> bool:
            """
            Perform Depth-First Search to check if the word exists in the Trie.
            - cur: Current node in the Trie
            - index: Current character index in the word being searched
            """
            # Base case: If we've reached the end of the word, check for the end-of-word marker
            if index == len(word):
                return '#' in cur

            # Get the current character
            char = word[index]

            if char == '.':
                # Wildcard case: Try all possible child nodes
                for key in cur:
                    # Skip the '#' marker and recursively search the next character
                    if key != '#' and dfs(cur[key], index + 1):
                        return True
            elif char in cur:
                # Regular character case: Check the specific child node
                if dfs(cur[char], index + 1):
                    return True
            
            # If no match is found, return False
            return False

        # Start DFS from the root of the Trie
        return dfs(self.root, 0)

```

## Find First and Last Position of Element in Sorted Array

```python
from typing import List

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # Helper function to perform binary search
        # If `first` is True, search for the first occurrence
        # If `first` is False, search for the last occurrence
        def binarySearch(nums: List[int], target: int, first: bool) -> int:
            left, right = 0, len(nums) - 1
            index = -1  # Default index when the target is not found
            
            while left <= right:
                mid = (left + right) // 2  # Calculate the middle index
                
                if nums[mid] < target:
                    left = mid + 1  # Move right if the target is larger
                elif nums[mid] > target:
                    right = mid - 1  # Move left if the target is smaller
                else:
                    # Target found, update index
                    index = mid
                    if first:
                        right = mid - 1  # Continue searching in the left part
                    else:
                        left = mid + 1  # Continue searching in the right part
            
            return index

        # Edge case: If the array is empty
        if not nums:
            return [-1, -1]

        # Perform binary search to find the first and last occurrences
        first = binarySearch(nums, target, first=True)
        last = binarySearch(nums, target, first=False)

        # Return the range as a list
        return [first, last]

```

## Word Search II

```python
from typing import List

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        def dfs(row: int, col: int, node: dict, path: List[str]) -> None:
            # Save the current character and mark the cell as visited
            char = board[row][col]
            board[row][col] = '#'  # Mark the cell to avoid revisiting during this path

            # Move to the next Trie node for the current character
            node = node[char]
            path.append(char)  # Add the character to the current path

            # If a complete word is found, add it to the result
            if '.' in node:
                result.add(''.join(path))  # Use a set to avoid duplicates

            # Explore all valid neighboring cells
            for x, y in directions:
                new_row, new_col = row + x, col + y
                if 0 <= new_row < m and 0 <= new_col < n:  # Check bounds
                    if board[new_row][new_col] in node:  # Continue DFS if character is in Trie
                        dfs(new_row, new_col, node, path)

            # Backtrack: Restore the cell and remove the last character from the path
            path.pop()
            board[row][col] = char

        # Directions for moving up, down, left, and right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Dimensions of the board
        m, n = len(board), len(board[0])

        # Build a Trie (prefix tree) from the list of words
        trie = {}
        for word in words:
            cur = trie
            for char in word:
                if char not in cur:
                    cur[char] = {}  # Create a new node for the character
                cur = cur[char]
            cur['.'] = True  # Mark the end of a word

        # Result set to store found words
        result = set()

        # Start DFS from every cell in the board
        for i in range(m):
            for j in range(n):
                if board[i][j] in trie:  # Start DFS only if the character is in the Trie
                    dfs(i, j, trie, [])

        # Return the result as a list
        return list(result)

```

## KMP
```python
from typing import List

class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        def getPrefix(pattern: str) -> List[int]:
            """
            Build the prefix table for the needle.
            prefix[i] represents the length of the longest prefix that is also a suffix
            for the substring pattern[0:i+1].
            """
            prefix = [0] * len(pattern)
            k = 0  # Length of the current longest prefix

            for i in range(1, len(pattern)):
                # Fall back until characters match or we reach the start
                while k > 0 and pattern[k] != pattern[i]:
                    k = prefix[k - 1]

                # If characters match, extend the prefix
                if pattern[k] == pattern[i]:
                    k += 1
                prefix[i] = k  # Update prefix table

            return prefix

        # Edge case: If needle is empty, return 0
        if not needle:
            return 0

        # Build the prefix table for the needle
        prefix = getPrefix(needle)
        k = 0  # Current position in needle

        # Traverse the haystack to find the needle
        for i, char in enumerate(haystack):
            # Fall back until characters match or we reach the start
            while k > 0 and needle[k] != char:
                k = prefix[k - 1]

            # If characters match, move to the next character in the needle
            if needle[k] == char:
                k += 1

            # If we've matched the entire needle, return the starting index
            if k == len(needle):
                return i - k + 1

        # If no match is found, return -1
        return -1

```

## Text Justification

```python
from typing import List

class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        result = []  # To store the justified lines
        line, length = [], 0  # Current line and its total length

        for word in words:
            # Check if adding the next word exceeds maxWidth
            if length + len(line) + len(word) > maxWidth:
                # Calculate the extra spaces to distribute
                extra_spaces = maxWidth - length
                if len(line) == 1:
                    # If there's only one word, left justify the line
                    result.append(line[0] + ' ' * extra_spaces)
                else:
                    # Distribute spaces evenly
                    spaces = extra_spaces // (len(line) - 1)
                    remainder = extra_spaces % (len(line) - 1)

                    # Add spaces to words in the line
                    for j in range(len(line) - 1):
                        line[j] += ' ' * spaces
                        if remainder > 0:
                            line[j] += ' '
                            remainder -= 1

                    # Join the line into a single string
                    result.append(''.join(line))

                # Reset the line and its length for the next line
                line, length = [], 0

            # Add the current word to the line
            line.append(word)
            length += len(word)

        # Handle the last line (left-justified)
        last_line = ' '.join(line)
        last_line += ' ' * (maxWidth - len(last_line))
        result.append(last_line)

        return result

```
