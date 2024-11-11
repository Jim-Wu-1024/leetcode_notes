#
# @lc app=leetcode id=518 lang=python3
#
# [518] Coin Change II
#

# @lc code=start
from typing import List

class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        # Initialize a DP array where dp[j] represents the number of ways to achieve amount j
        dp = [0] * (amount + 1)
        dp[0] = 1  # There is one way to make amount 0: use no coins
        
        # Iterate over each coin in coins
        for coin in coins:
            # Update the dp array from the value of the coin to the target amount
            for j in range(coin, amount + 1):
                # Add the ways to achieve amount j - coin to dp[j]
                dp[j] += dp[j - coin]
        
        # The final answer is the number of ways to make up the 'amount'
        return dp[amount]
    
# @lc code=end

