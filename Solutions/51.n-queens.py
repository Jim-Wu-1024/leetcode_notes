#
# @lc app=leetcode id=51 lang=python3
#
# [51] N-Queens
#

# @lc code=start
from typing import List

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def isValid(row: int, col: int, chessboard: List[List[str]]) -> bool:
            for i in range(row):
                if chessboard[i][col] == 'Q':
                    return False
            for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
                if chessboard[i][j] == 'Q':
                    return False
            for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
                if chessboard[i][j] == 'Q':
                    return False
            return True

        def backtracing(row: int, chessboard: List[List[str]]):
            if row == n:
                result.append([''.join(r) for r in chessboard])
                return

            for i in range(n):
                if isValid(row, i, chessboard):
                    chessboard[row][i] = 'Q'
                    backtracing(row+1, chessboard)
                    chessboard[row][i] = '.'

        result = []
        chessboard = [['.' for _ in range(n)] for _ in range(n)]
        backtracing(0, chessboard)
        return result
          
# @lc code=end

