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