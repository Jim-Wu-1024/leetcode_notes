#
# @lc app=leetcode id=47 lang=python3
#
# [47] Permutations II
#

# @lc code=start
from typing import List

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def backtracing(path: List[int], used: List[bool]):
            if len(path) == len(nums):
                result.append(path.copy())
                return
            
            for i in range(len(nums)):
                if used[i]:
                    continue

                if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                    continue

                path.append(nums[i])
                used[i] = True
                backtracing(path, used)
                used[i] = False
                path.pop()
        
        nums.sort()
        result = []
        backtracing([], [False]*len(nums))
        return result
    
# @lc code=end

