#
# @lc app=leetcode id=45 lang=python3
#
# [45] Jump Game II
#

# @lc code=start
from typing import List

class Solution:
    def jump(self, nums: List[int]) -> int:
        # If the array has one or no elements, no jumps are needed
        if len(nums) <= 1:
            return 0

        cur_max_reach, next_max_reach = 0, 0
        count = 0
        for i in range(len(nums)):
            next_max_reach = max(next_max_reach, i+nums[i])

            if i == cur_max_reach:
                # When we reach the end of the current jump's reach
                count += 1
                cur_max_reach = next_max_reach

                if cur_max_reach >= len(nums)-1:
                    break

        return count
        
# @lc code=end

