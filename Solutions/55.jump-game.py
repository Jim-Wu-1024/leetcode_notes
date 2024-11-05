#
# @lc app=leetcode id=55 lang=python3
#
# [55] Jump Game
#

# @lc code=start
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        # Corrected the empty list check
        if not nums:
            return True
        
        max_reach = 0  # Renamed for clarity
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
    
# @lc code=end

