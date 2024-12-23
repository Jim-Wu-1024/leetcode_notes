#
# @lc app=leetcode id=674 lang=python3
#
# [674] Longest Continuous Increasing Subsequence
#

# @lc code=start
from typing import List

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
     
# @lc code=end

