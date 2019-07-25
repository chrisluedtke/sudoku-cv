import copy
from typing import List


class Sudoku:
    def __init__(self, board: List[List[str]], values: List[str], blank='.'):
        # TODO: validate board
        self.board = copy.deepcopy(board)
        self.blank = blank
        self.val_set = set(values)  # set of all values that may be placed
        self.solution = self.solve()

    def solve(self):
        board = copy.deepcopy(self.board)

        d = {}  # create a dict of slot locations to fill
        i = 0
        for r_i in range(9):
            for c_i in range(9):
                if board[r_i][c_i] == self.blank:
                    d[i] = (r_i, c_i)
                    i += 1

        # Iterate over slots, inserting values that are valid.
        # If no valid value possible, backtrack to previous slot
        #   and increment by 1
        i = 0
        while i < len(d):  # board is solved if last empty slot is filled
            if i < 0:  # failure case, board has no solution
                return None
            # look up the current row/column slot location from the dict
            r_i = d[i][0]
            c_i = d[i][1]

            # get set of values already used in row/col/box constraints
            row_vals = set(board[r_i])
            col_vals = set(board[i][c_i] for i in range(9))
            b_r_i = r_i // 3 * 3
            b_c_i = c_i // 3 * 3
            box_vals = set(
                board[b_r_i    ][b_c_i:b_c_i + 3] + 
                board[b_r_i + 1][b_c_i:b_c_i + 3] + 
                board[b_r_i + 2][b_c_i:b_c_i + 3]
            )

            # subtract already-used numbers from the set of all values
            candidates = self.val_set - row_vals - col_vals - box_vals
            # filter candidates to those greater than current placement
            cur_val = board[r_i][c_i]
            # TODO: this line assumes self.blank is less than all values
            candidates = sorted(c for c in candidates if c > cur_val)

            if not candidates:  # backtrack if no valid candidates exist
                board[r_i][c_i] = self.blank
                i -= 1

            else:  # otherwise place the next greater candidate in the slot
                board[r_i][c_i] = candidates[0]
                i += 1

        return board                   
