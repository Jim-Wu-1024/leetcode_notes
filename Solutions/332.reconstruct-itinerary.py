#
# @lc app=leetcode id=332 lang=python3
#
# [332] Reconstruct Itinerary
#

# @lc code=start
from typing import List
from collections import defaultdict

class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        def dfs(node: str):
            while path[node]:
                dfs(path[node].pop(0))
            result.append(node)

        
        path = defaultdict(list)
        for ticket in tickets:
            path[ticket[0]].append(ticket[1])
        for node in path.keys():
            path[node].sort()

        result = []
        dfs('JFK')
        return result[::-1]

        
# @lc code=end

