#
# @lc app=leetcode id=40 lang=python3
#
# [40] Combination Sum II
#

# @lc code=start
from typing import List

class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtracking(start: int, path: List[int], currentSum: int):
            if currentSum == target:
                result.append(path.copy())
                return
            if currentSum > target:
                return
            
            for i in range(start, len(candidates)):
                # If adding candidates[i] would exceed the target, exit the loop
                if currentSum + candidates[i] > target:
                    break
                # Skip duplicate elements to avoid duplicate combinations
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                
                path.append(candidates[i])
                backtracking(i + 1, path, currentSum + candidates[i])
                path.pop()  # Backtrack to try the next candidate
        
        if not candidates or min(candidates) > target:
            return []
        
        result = []
        candidates.sort()
        backtracking(0, [], 0)
        return result
        
# @lc code=end

