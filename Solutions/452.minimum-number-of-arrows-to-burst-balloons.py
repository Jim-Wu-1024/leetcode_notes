#
# @lc app=leetcode id=452 lang=python3
#
# [452] Minimum Number of Arrows to Burst Balloons
#

# @lc code=start
from typing import List

class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda x: x[0])

        count = 1 
        for i in range(1, len(points)):
            if points[i][0] > points[i-1][1]:
                count += 1
            else:
                points[i][1] = min(points[i-1][1], points[i][1])

        return count
        
# @lc code=end

