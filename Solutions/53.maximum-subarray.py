#
# @lc app=leetcode id=53 lang=python3
#
# [53] Maximum Subarray
#

# @lc code=start
from typing import List

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
    
# @lc code=end

