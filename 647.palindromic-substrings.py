#
# @lc app=leetcode id=647 lang=python3
#
# [647] Palindromic Substrings
#

# @lc code=start
class Solution:
    def countSubstrings(self, s: str) -> int:
        def expandAroundCenter(left: int, right: int) -> int:
            count = 0
            # Expand outward while the characters match and stay within bounds
            while left >= 0 and right < len(s) and s[left] == s[right]:
                count += 1  # Found a palindrome
                left -= 1   # Move left pointer outward
                right += 1  # Move right pointer outward
            return count
        
        result = 0
        for i in range(len(s)):
            # Count odd-length palindromes (single character as center)
            result += expandAroundCenter(i, i)
            # Count even-length palindromes (two characters as center)
            result += expandAroundCenter(i, i + 1)
        
        return result
    
# @lc code=end

