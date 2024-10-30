#
# @lc app=leetcode id=491 lang=python3
#
# [491] Non-decreasing Subsequences
#

# @lc code=start
from typing import List

class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        def backtracing(start: int, path: List[int]):
            if len(path) >= 2:
                result.append(path.copy())

            if start >= len(nums):
                return
            
            used = {}
            for i in range(start, len(nums)):
                if (len(path) > 0 and nums[i] < path[-1]) or nums[i] in used:
                    continue

                path.append(nums[i])
                used[nums[i]] = 1

                backtracing(i+1, path)

                path.pop()

            
        result = []
        backtracing(0, [])
        return result
        
# @lc code=end

