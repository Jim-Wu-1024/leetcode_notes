#
# @lc app=leetcode id=93 lang=python3
#
# [93] Restore IP Addresses
#

# @lc code=start
from typing import List

class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def isValid(left: int, right: int) -> bool:
            segment = s[left:right+1]
            # Check if segment is a digit-only string
            if not segment.isdigit():
                return False
            # Check for leading zero in multi-digit segment
            if len(segment) > 1 and segment[0] == "0":
                return False
            # Check if segment is within the valid IP range
            if int(segment) > 255:
                return False
            return True

        def backtracking(start: int, path: List[str]):
            if len(path) == 4 and start == len(s):
                result.append(".".join(path))
                return 
            if len(path) >= 4:
                return
            
            for i  in range(start, min(start + 3, len(s))):
                if len(path) > 4:
                    break
                if isValid(start, i):
                    path.append(s[start:i+1])
                    backtracking(i+1, path)
                    path.pop()

        result = []
        backtracking(0, [])
        return result        

        
# @lc code=end