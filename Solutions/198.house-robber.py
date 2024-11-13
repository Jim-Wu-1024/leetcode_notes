#
# @lc app=leetcode id=198 lang=python3
#
# [198] House Robber
#

# @lc code=start
from typing import List

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
    
# @lc code=end

