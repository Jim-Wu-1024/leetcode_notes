#
# @lc app=leetcode id=746 lang=python3
#
# [746] Min Cost Climbing Stairs
#

# @lc code=start
from typing import List

class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # If there are only two steps, return the minimum of the two, as that's the minimum cost to reach the top
        if len(cost) == 2:
            return min(cost)
        
        # Initialize a DP array where dp[i] represents the minimum cost to reach step i
        dp = [0] * len(cost)
        dp[0], dp[1] = cost[0], cost[1]
        
        # Fill the DP array with the minimum cost to reach each step starting from step 2
        for i in range(2, len(cost)):
            # The cost to reach step i is the minimum of reaching it from i-1 or i-2
            dp[i] = min(dp[i - 1] + cost[i], dp[i - 2] + cost[i])
        
        # The result is the minimum cost of reaching the top, which can be done either from the last step
        # or the second-to-last step
        return min(dp[-1], dp[-2])
    
# @lc code=end

