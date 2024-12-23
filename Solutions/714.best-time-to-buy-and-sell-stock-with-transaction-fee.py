#
# @lc app=leetcode id=714 lang=python3
#
# [714] Best Time to Buy and Sell Stock with Transaction Fee
#

# @lc code=start
from typing import List

class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        # Edge case: if only one price is given, no transactions can be made
        if len(prices) == 1:
            return 0
        
        # Initialize DP array where:
        # dp[i][0] represents the max profit on day i without holding any stock.
        # dp[i][1] represents the max profit on day i while holding a stock.
        dp = [[0, 0] for _ in range(len(prices))]
        
        # Base cases
        dp[0][0] = 0            # No stock held on day 0, profit is 0
        dp[0][1] = -prices[0]    # Stock bought on day 0, initial negative profit
        
        # Populate DP array for each subsequent day
        for i in range(1, len(prices)):
            # dp[i][0]: Max profit on day i without holding any stock
            # Choices: do nothing (dp[i-1][0]) or sell stock (dp[i-1][1] + prices[i] - fee)
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i] - fee)
            
            # dp[i][1]: Max profit on day i while holding a stock
            # Choices: keep holding (dp[i-1][1]) or buy stock (dp[i-1][0] - prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])

        # The maximum profit achievable by the last day without holding stock is in dp[-1][0]
        return dp[-1][0]
    
# @lc code=end

