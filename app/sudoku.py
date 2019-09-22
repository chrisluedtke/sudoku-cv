import copy
from typing import List

import numpy as np
from skimage.draw import polygon_perimeter


class Sudoku:
    board_coords = np.array([])
    board = []
    blank = 0
    val_set = set(range(1, 10))
    solution = None

    @classmethod
    def solve(cls):
        assert len(cls.board) == 9 and len(cls.board[0]) == 9

        board = copy.deepcopy(cls.board)

        d = {}  # create a dict of slot locations to fill
        i = 0
        for r_i in range(9): 
            for c_i in range(9):
                if board[r_i][c_i] == cls.blank:
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
            candidates = cls.val_set - row_vals - col_vals - box_vals
            # filter candidates to those greater than current placement
            cur_val = board[r_i][c_i]
            # this line assumes cls.blank is less than all values
            candidates = sorted(c for c in candidates if c > cur_val)

            if not candidates:  # backtrack if no valid candidates exist
                board[r_i][c_i] = cls.blank
                i -= 1

            else:  # otherwise place the next greater candidate in the slot
                board[r_i][c_i] = candidates[0]
                i += 1

        return board

    @classmethod
    def set_board_coords(cls, coords):
        cls.board_coords = coords

    @classmethod
    def get_board_coords(cls):
        return cls.board_coords

    @classmethod
    def __repr__(cls):
        return cls.board

    @staticmethod
    def process_frame(img):
        if Sudoku.board_coords.size:  # Draw board border
            rr, cc = polygon_perimeter(
                Sudoku.board_coords[:, 1], 
                Sudoku.board_coords[:, 0], 
                shape=img.shape)
            img[rr, cc] = (0, 255, 0)

        return img


if __name__ == "__main__":
    board = [["5", "3",  "",  "", "7",  "",  "",  "",  ""],
             ["6",  "",  "", "1", "9", "5",  "",  "",  ""],
             [ "", "9", "8",  "",  "",  "",  "", "6",  ""],
             ["8",  "",  "",  "", "6",  "",  "",  "", "3"],
             ["4",  "",  "", "8",  "", "3",  "",  "", "1"],
             ["7",  "",  "",  "", "2",  "",  "",  "", "6"],
             [ "", "6",  "",  "",  "",  "", "2", "8",  ""],
             [ "",  "",  "", "4", "1", "9",  "",  "", "5"],
             [ "",  "",  "",  "", "8",  "",  "", "7", "9"]]

    s = Sudoku(board,
               values=[str(i) for i in range(1, 10)],
               blank="")
