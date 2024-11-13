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