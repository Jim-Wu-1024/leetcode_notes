#
# @lc app=leetcode id=53 lang=python3
#
# [53] Maximum Subarray
#

# @lc code=start
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        preSum = nums[0]
        maxSum = nums[0]

        for i in range(1, len(nums)):
            preSum = nums[i] if preSum < 0 else preSum + nums[i]
            
            if preSum > maxSum:
                maxSum = preSum
                
        return maxSum
        
# @lc code=end

 