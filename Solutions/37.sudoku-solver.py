#
# @lc app=leetcode id=37 lang=python3
#
# [37] Sudoku Solver
#

# @lc code=start
from typing import List

class Solution:
     def solveSudoku(self, board: List[List[str]]) -> None:
        def isValid(rowNum: int, colNum: int, num: str, board: List[List[str]]) -> bool:
        # Check the row and column
            for i in range(9):
                if board[rowNum][i] == num or board[i][colNum] == num:
                    return False

            # Determine the starting indices of the 3x3 subgrid
            rowStart = (rowNum // 3) * 3
            colStart = (colNum // 3) * 3

            # Check the 3x3 subgrid
            for i in range(rowStart, rowStart + 3):
                for j in range(colStart, colStart + 3):
                    if board[i][j] == num:
                        return False

            return True
        
        def backtracing(board: List[List[str]]) -> bool:
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        for num in range(1, 10):
                            if isValid(i, j, str(num), board):
                                board[i][j] = str(num)
                                if backtracing(board):
                                    return True
                                board[i][j] = '.'  # Backtrack
                        return False  # If no valid number, return False
            return True  # All cells are filled correctly

        backtracing(board)  # Start the backtracking process

        
# @lc code=end

