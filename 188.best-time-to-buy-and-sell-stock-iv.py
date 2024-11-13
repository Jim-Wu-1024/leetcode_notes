#
# @lc app=leetcode id=188 lang=python3
#
# [188] Best Time to Buy and Sell Stock IV
#

# @lc code=start
from typing import List

class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if len(prices) == 1:
            return 0
        
        dp = [[0] * (2 * k)  for _ in range(len(prices))]
        for i in range(2 * k):
            if i % 2 == 0:
                dp[0][i] = 0
            else:
                dp[0][i] = -prices[0]

        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], -prices[i])

            for j in range(2, 2 * k, 2):
                dp[i][j] = max(dp[i-1][j], dp[i-1][j+1] + prices[i])
                dp[i][j+1] = max(dp[i-1][j+1], dp[i-1][j+1-3] - prices[i])

        return dp[-1][-2]

        
# @lc code=end

