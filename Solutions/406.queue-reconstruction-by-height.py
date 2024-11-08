#
# @lc app=leetcode id=406 lang=python3
#
# [406] Queue Reconstruction by Height
#

# @lc code=start
from typing import List

class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        if len(people) == 1:
            return people
        
        people.sort(key=lambda x: (-x[0], x[1]))
        
        new_queue = []
        for one in people:
            pos = one[1]

            new_queue.insert(pos, one)
        return new_queue

# @lc code=end

