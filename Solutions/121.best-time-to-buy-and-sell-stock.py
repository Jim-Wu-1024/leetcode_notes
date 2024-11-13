#
# @lc app=leetcode id=121 lang=python3
#
# [121] Best Time to Buy and Sell Stock
#

# @lc code=start
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # Edge case: if there is only one day, no transactions can be made
        if len(prices) == 1:
            return 0
        
        # Initialize the DP array where dp[i][0] is the max profit on day i with no stock
        # and dp[i][1] is the max profit on day i with one stock (bought at some point)
        dp = [[0, 0] for _ in range(len(prices))]
        
        # Base cases
        dp[0][0] = 0           # If we do not buy on the first day, profit is 0
        dp[0][1] = -prices[0]  # If we buy on the first day, profit is -prices[0]
        
        # Fill the DP array for each day
        for i in range(1, len(prices)):
            # dp[i][0]: max profit if we do not hold stock on day i
            # Choices: not selling (keep dp[i-1][0]) or sell stock bought on an earlier day (dp[i-1][1] + prices[i])
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            
            # dp[i][1]: max profit if we hold stock on day i
            # Choices: keep holding (dp[i-1][1]) or buy stock today (since only one transaction is allowed, set to -prices[i])
            dp[i][1] = max(dp[i-1][1], -prices[i])

        # The maximum profit achievable is without holding stock on the last day (dp[-1][0])
        return dp[-1][0]

# @lc code=end

