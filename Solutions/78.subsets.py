#
# @lc app=leetcode id=78 lang=python3
#
# [78] Subsets
#

# @lc code=start
from typing import List

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtracking(start: int, path: List[int]):
            result.append(path[:])
            
            if start >= len(nums):
                return
            
            for i in range(start, len(nums)):
                path.append(nums[i])
                backtracking(i+1, path)
                path.pop()

        result = []
        backtracking(0, [])
        return result
        
# @lc code=end