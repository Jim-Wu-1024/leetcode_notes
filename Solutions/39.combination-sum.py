#
# @lc app=leetcode id=39 lang=python3
#
# [39] Combination Sum
#

# @lc code=start
from typing import List

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtracing(start: int, path: List[int], currentSum: int):
            if currentSum == target:
                result.append(path[:])
                return
            if currentSum > target:
                return
            
            for i in range(start, len(candidates)):
                if currentSum + candidates[i] > target:
                    break
                path.append(candidates[i])
                backtracing(i, path, currentSum+candidates[i])
                path.pop()

        result = []
        candidates.sort()
        backtracing(0, [], 0)
        return result
        
# @lc code=end

