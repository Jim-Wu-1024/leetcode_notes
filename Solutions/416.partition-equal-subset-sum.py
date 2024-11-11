#
# @lc app=leetcode id=416 lang=python3
#
# [416] Partition Equal Subset Sum
#

# @lc code=start
from typing import List

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        target = sum(nums)
        
        if target % 2 == 1:
            return False
        
        target //= 2
        dp = [0] * (target+1)
        for num in nums:
            for j in range(target, num-1 ,-1):
                dp[j] = max(dp[j], dp[j-num] + num)

        return dp[target] == target    
    
# @lc code=end

