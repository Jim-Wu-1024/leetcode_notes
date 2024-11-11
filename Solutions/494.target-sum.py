#
# @lc app=leetcode id=494 lang=python3
#
# [494] Target Sum
#

# @lc code=start
from typing import List

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # Calculate the sum of all numbers
        total_sum = sum(nums)
        
        # Check if it’s possible to achieve the target
        # If total_sum + target is odd or if target is too large to be achieved, return 0
        if (total_sum + target) % 2 == 1 or total_sum < abs(target):
            return 0
        
        # Calculate the subset sum we need to achieve
        subset_sum = (total_sum + target) // 2
        
        # Initialize dp array, where dp[j] will store the number of ways to achieve sum j
        dp = [0] * (subset_sum + 1)
        dp[0] = 1  # There's one way to make zero sum: choose no elements
        
        # Fill the dp array by iterating through each number in nums
        for num in nums:
            # Traverse backwards to avoid reusing the same number in the same iteration
            for j in range(subset_sum, num - 1, -1):
                dp[j] += dp[j - num]
        
        # The answer is the number of ways to achieve the subset_sum
        return dp[subset_sum]
          
# @lc code=end

