#
# @lc app=leetcode id=377 lang=python3
#
# [377] Combination Sum IV
#

# @lc code=start
from typing import List

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        # Initialize the DP array where dp[j] represents the number of ways to reach sum j
        dp = [0] * (target + 1)
        dp[0] = 1  # There is one way to achieve the sum 0: using no elements
        
        # Iterate over each possible sum from 1 up to the target
        for j in range(1, target + 1):
            # Check each number in nums to see if it can contribute to the current sum j
            for num in nums:
                # If num can be used to reach sum j (i.e., j >= num)
                if j >= num:
                    # Add the number of ways to achieve the sum (j - num) to dp[j]
                    dp[j] += dp[j - num]
        
        # The answer is the number of ways to achieve the target sum
        return dp[target]
    
# @lc code=end

