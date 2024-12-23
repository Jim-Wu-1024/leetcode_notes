#
# @lc app=leetcode id=80 lang=python3
#
# [80] Remove Duplicates from Sorted Array II
#

# @lc code=start
from typing import List

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
    
# @lc code=end

