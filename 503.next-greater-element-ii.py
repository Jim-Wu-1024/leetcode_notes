#
# @lc app=leetcode id=503 lang=python3
#
# [503] Next Greater Element II
#

# @lc code=start
from typing import List

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
    
# @lc code=end

