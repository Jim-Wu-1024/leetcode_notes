#
# @lc app=leetcode id=376 lang=python3
#
# [376] Wiggle Subsequence
#

# @lc code=start
from typing import List

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

# @lc code=end

