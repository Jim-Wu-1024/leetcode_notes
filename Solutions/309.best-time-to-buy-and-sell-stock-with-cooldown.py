#
# @lc app=leetcode id=309 lang=python3
#
# [309] Best Time to Buy and Sell Stock with Cooldown
#

# @lc code=start
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # Edge case: if there is only one price, no transactions can be made
        if len(prices) == 1:
            return 0
        
        # Initialize DP array where:
        # dp[i][0] represents the max profit on day i without holding any stock.
        # dp[i][1] represents the max profit on day i while holding a stock.
        dp = [[0, 0] for _ in range(len(prices))]
        
        # Base cases
        dp[0][0] = 0            # No stock held on day 0, profit is 0
        dp[0][1] = -prices[0]    # Stock bought on day 0, negative profit by price of the stock
        dp[1][0] = max(dp[0][0], dp[0][1] + prices[1])  # Sell the stock on day 1 or do nothing
        dp[1][1] = max(dp[0][1], -prices[1])            # Buy the stock on day 1 or hold from day 0
        
        # Populate the DP array for each day from day 2 onwards
        for i in range(2, len(prices)):
            # dp[i][0]: Max profit on day i without holding any stock
            # Choices: do nothing (dp[i-1][0]) or sell stock held from previous day (dp[i-1][1] + prices[i])
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            
            # dp[i][1]: Max profit on day i while holding a stock
            # Choices: keep holding (dp[i-1][1]) or buy stock today (dp[i-2][0] - prices[i] due to cooldown)
            dp[i][1] = max(dp[i-1][1], dp[i-2][0] - prices[i])

        print(dp)
        # Return the max profit on the last day without holding any stock
        return dp[-1][0]
    
# @lc code=end

