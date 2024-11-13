#
# @lc app=leetcode id=122 lang=python3
#
# [122] Best Time to Buy and Sell Stock II
#

# @lc code=start
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # Edge case: if there's only one day, no transactions can be made
        if len(prices) == 1:
            return 0
        
        # Initialize the DP array:
        # dp[i][0]: max profit on day i with no stock held
        # dp[i][1]: max profit on day i with stock held
        dp = [[0, 0] for _ in range(len(prices))]
        
        # Base cases
        dp[0][0] = 0               # No stock on the first day means no profit
        dp[0][1] = -prices[0]      # Buying stock on the first day costs prices[0]
        
        # Fill in the DP array for each subsequent day
        for i in range(1, len(prices)):
            # dp[i][0]: max profit if no stock is held on day i
            # Choices: do nothing (dp[i-1][0]) or sell stock bought on an earlier day (dp[i-1][1] + prices[i])
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            
            # dp[i][1]: max profit if stock is held on day i
            # Choices: keep holding stock (dp[i-1][1]) or buy stock today (dp[i-1][0] - prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])

        # Return the max profit at the end of the last day with no stock held
        return dp[-1][0]
        
# @lc code=end

