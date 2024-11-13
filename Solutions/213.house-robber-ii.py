#
# @lc app=leetcode id=213 lang=python3
#
# [213] House Robber II
#

# @lc code=start
from typing import List

class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return 0
        
        numsHead = nums[0:-1]
        numsTail = nums[1:]

        if len(numsHead) == 2 and len(numsTail) == 2:
            return max(max(numsHead[0], numsHead[1]), max(numsTail[0], numsTail[1]))

        dpHead = [0] * len(numsHead)
        dpHead[0] = numsHead[0]
        dpHead[1] = max(numsHead[0], numsHead[1])

        dpTail = [0] * len(numsTail)
        dpTail[0] = numsTail[0]
        dpTail[1] = max(numsTail[0], numsTail[1])

        for i in range(2, len(numsHead)):
            dpHead[i] = max(dpHead[i-1], dpHead[i-2] + numsHead[i])
            dpTail[i] = max(dpTail[i-1], dpTail[i-2] + numsTail[i])

        return max(dpHead[-1], dpTail[-1])
        
# @lc code=end

