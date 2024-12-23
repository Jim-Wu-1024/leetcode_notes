#
# @lc app=leetcode id=300 lang=python3
#
# [300] Longest Increasing Subsequence
#

# @lc code=start
from typing import List

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
            
# @lc code=end

