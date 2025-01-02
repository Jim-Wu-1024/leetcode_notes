#
# @lc app=leetcode id=26 lang=python3
#
# [26] Remove Duplicates from Sorted Array
#

# @lc code=start
from typing import List

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
    
# @lc code=end

