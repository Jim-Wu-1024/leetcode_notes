#
# @lc app=leetcode id=123 lang=python3
#
# [123] Best Time to Buy and Sell Stock III
#

# @lc code=start
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # Edge case: If only one price is given, return 0 as no transactions can be made
        if len(prices) == 1:
            return 0
        
        # Initialize DP array:
        # dp[i][0]: max profit with no transactions up to day i
        # dp[i][1]: max profit with first buy up to day i
        # dp[i][2]: max profit with one complete transaction (buy + sell) up to day i
        # dp[i][3]: max profit with second buy up to day i
        dp = [[0, 0, 0, 0] for _ in range(len(prices))]
        dp[0][0], dp[0][1], dp[0][2], dp[0][3] = 0, -prices[0], 0, -prices[0]

        # Populate the DP array for each day
        for i in range(1, len(prices)):
            # dp[i][0]: Max profit up to day i with no transactions (do nothing)
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            
            # dp[i][1]: Max profit after the first buy (buy stock on day i or do nothing)
            dp[i][1] = max(dp[i-1][1], -prices[i])
            
            # dp[i][2]: Max profit after completing the first transaction
            # (sell stock bought on or before day i or do nothing)
            dp[i][2] = max(dp[i-1][2], dp[i-1][3] + prices[i])
            
            # dp[i][3]: Max profit after the second buy (buy stock on day i after the first sell)
            dp[i][3] = max(dp[i-1][3], dp[i-1][0] - prices[i])

        # Return the maximum profit after completing up to two transactions (dp[-1][2])
        return dp[-1][2]
    
# @lc code=end
