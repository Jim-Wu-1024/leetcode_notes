#
# @lc app=leetcode id=322 lang=python3
#
# [322] Coin Change
#

# @lc code=start
from typing import List

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # Initialize the DP array with infinity, representing an initially unreachable amount
        dp = [float('inf')] * (amount + 1)
        
        # Base case: 0 coins are needed to make amount 0
        dp[0] = 0
        
        # Loop through each coin in the coins list
        for coin in coins:
            # For each coin, update the dp array for amounts from `coin` to `amount`
            for j in range(coin, amount + 1):
                # Update dp[j] to be the minimum of its current value or
                # the value of dp[j - coin] + 1 (adding this coin to the minimum solution for `j - coin`)
                dp[j] = min(dp[j], dp[j - coin] + 1)
        
        # If dp[amount] is still infinity, it means the amount cannot be made with the given coins
        if dp[amount] == float('inf'):
            return -1
        
        # Otherwise, return the minimum number of coins needed for `amount`
        return dp[amount]
    
# @lc code=end

