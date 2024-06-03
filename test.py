import unittest
from sprites import *

class TestBoardMethods(unittest.TestCase):
    def setUp(self):
        self.board = Board()

    def test_get_neighbours(self):
        # Choose a tile in the middle of the board
        x, y = ROWS // 2, COLS // 2
        print(f"Tile coordinates: ({x}, {y})")  # Output the coordinates of the tile

        # Get the neighbors of the tile
        neighbors = self.board.get_neighbours(x, y)

        # Check if the correct number of neighbors is returned
        self.assertEqual(len(neighbors), 8)

        # Check if each neighbor is adjacent to the tile
        for neighbor in neighbors:
            print(f"Neighbor coordinates: ({neighbor.x // TILESIZE}, {neighbor.y // TILESIZE})")  # Output the coordinates of the neighbor
            self.assertTrue(abs(neighbor.x - x * TILESIZE) <= TILESIZE and abs(neighbor.y - y * TILESIZE) <= TILESIZE)

if __name__ == '__main__':
    unittest.main()