#
# @lc app=leetcode id=134 lang=python3
#
# [134] Gas Station
#

# @lc code=start
from typing import List

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        cur_sum, total_sum = 0, 0
        start = 0

        for i in range(len(gas)):
            gas_gain = gas[i] - cost[i]
            cur_sum += gas_gain
            total_sum += gas_gain

            if cur_sum < 0:
                start = i + 1
                cur_sum = 0

        if total_sum < 0:
            return -1
        return start
        
# @lc code=end

