#
# @lc app=leetcode id=435 lang=python3
#
# [435] Non-overlapping Intervals
#

# @lc code=start
from typing import List

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if len(intervals) == 1:
            return 0

        intervals.sort(key=lambda x: x[0])

        count = 0
        for i in range(1, len(intervals)):
            if intervals[i-1][1] > intervals[i][0]:
                count += 1
                intervals[i][1] = min(intervals[i-1][1], intervals[i][1])

        return count
        
# @lc code=end

