#
# @lc app=leetcode id=46 lang=python3
#
# [46] Permutations
#

# @lc code=start
from typing import List, Dict

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(path: List[int], used: List[bool]):
            # Base case: if path length is the same as nums, we have a complete permutation
            if len(path) == len(nums):
                result.append(path[:])
                return
            
            for i in range(len(nums)):
                # Skip used elements to avoid reusing the same element in the same permutation
                if used[i]:
                    continue
                
                # Add current element to path and mark as used
                path.append(nums[i])
                used[i] = True
                
                # Recurse with updated path and used list
                backtrack(path, used)
                
                # Backtrack: remove current element from path and mark as unused
                path.pop()
                used[i] = False

        result = []
        backtrack([], [False] * len(nums))
        return result
        
# @lc code=end

