import random

import pygame
from settings import *

# types list
# "." -> unknown
# "X" -> mine
# "C" -> clue
# "/" -> empty


class Tile:
    def __init__(self, x, y, image, type, number=None, revealed=False, flagged=False, color=None):
        self.x, self.y = x * TILESIZE, y * TILESIZE
        self.image = image
        self.type = type
        self.number = number
        self.revealed = revealed
        self.flagged = flagged
        self.color = color

    def draw(self, board_surface):
        if self.color:
            self.image.fill(self.color)
        if not self.flagged and self.revealed:
            board_surface.blit(self.image, (self.x, self.y))
        elif self.flagged and not self.revealed:
            board_surface.blit(tile_flag, (self.x, self.y))
        elif not self.revealed:
            board_surface.blit(tile_unknown, (self.x, self.y))

    def __lt__(self, other):
        if isinstance(other, Tile):
            return (self.x, self.y) < (other.x, other.y)

    def __repr__(self):
        return f"Tile(type={self.type}, x={self.x // TILESIZE}, y={self.y // TILESIZE}, number={self.number}, color={self.color})"


class Board:
    def __init__(self):
        self.board_surface = pygame.Surface((WIDTH, HEIGHT))
        self.board_list = [[Tile(row, col, tile_empty, ".") for col in range(COLS)] for row in range(ROWS)]
        self.place_mines()
        self.place_clues()
        self.dug = []

    def place_mines(self):
        for _ in range(AMOUNT_MINES):
            while True:
                row = random.randint(0, ROWS-1)
                col = random.randint(0, COLS-1)

                if self.board_list[row][col].type == ".":
                    self.board_list[row][col].image = tile_mine
                    self.board_list[row][col].type = "X"
                    break

    def place_clues(self):
        for x in range(ROWS):
            for y in range(COLS):
                if self.board_list[x][y].type != "X":
                    total_mines = self.check_neighbours(x, y)
                    if total_mines > 0:
                        self.board_list[x][y].image = tile_numbers[total_mines - 1]
                        self.board_list[x][y].type = "C"
                        self.board_list[x][y].number = total_mines  # Add this line

    @staticmethod
    def is_inside(x, y):
        return 0 <= x < ROWS and 0 <= y < COLS
    
    def get_covered_tiles(self):
        covered_tiles = []
        for row in self.board_list:
            for tile in row:
                if not tile.revealed and not tile.flagged:
                    covered_tiles.append(tile)
        return covered_tiles
    
    def calculate_remaining_mines(self):
        remaining_mines = AMOUNT_MINES
        for row in self.board_list:
            for tile in row:
                if tile.flagged:
                    remaining_mines -= 1
        return remaining_mines

    def check_neighbours(self, x, y):
        total_mines = 0
        for x_offset in range(-1, 2):
            for y_offset in range(-1, 2):
                neighbour_x = x + x_offset
                neighbour_y = y + y_offset
                if self.is_inside(neighbour_x, neighbour_y) and self.board_list[neighbour_x][neighbour_y].type == "X":
                    total_mines += 1

        return total_mines
    
    def is_boundary(self, tile):
        x, y = tile.x // TILESIZE, tile.y // TILESIZE
        if not self.board_list[x][y].revealed:
            neighbours = self.get_neighbours(x, y)
            for neighbour in neighbours:
                if neighbour.revealed:
                    return True
        return False
    
    def get_neighbours(self, x, y):
        neighbors = []
        for x_offset in range(-1, 2):
            for y_offset in range(-1, 2):
                neighbor_x = x + x_offset
                neighbor_y = y + y_offset
                if self.is_inside(neighbor_x, neighbor_y) and not (x_offset == 0 and y_offset == 0):
                    neighbors.append(self.board_list[neighbor_x][neighbor_y])
        return neighbors
    
    def cell_surroundings(self, tile):
        neighbors = []
        x, y = tile.x // TILESIZE, tile.y // TILESIZE
        for x_offset in range(-1, 2):
            for y_offset in range(-1, 2):
                neighbor_x = x + x_offset
                neighbor_y = y + y_offset
                if self.is_inside(neighbor_x, neighbor_y) and not (x_offset == 0 and y_offset == 0):
                    neighbors.append(self.board_list[neighbor_x][neighbor_y])
        return neighbors
    
    def get_random_unrevealed_tile(self):
        unrevealed_tiles = [(x, y) for x in range(ROWS) for y in range(COLS) if not self.board_list[x][y].revealed and not self.board_list[x][y].flagged]
        x, y = random.choice(unrevealed_tiles)
        return [self.board_list[x][y]]

    def draw(self, screen):
        for row in self.board_list:
            for tile in row:
                tile.draw(self.board_surface)
        screen.blit(self.board_surface, (0, 0))

    def dig(self, x, y):
        self.dug.append((x, y))
        if self.board_list[x][y].type == "X":
            self.board_list[x][y].revealed = True
            self.board_list[x][y].image = tile_exploded
            return False
        elif self.board_list[x][y].type == "C":
            self.board_list[x][y].revealed = True
            return True

        self.board_list[x][y].revealed = True

        for row in range(max(0, x-1), min(ROWS-1, x+1) + 1):
            for col in range(max(0, y-1), min(COLS-1, y+1) + 1):
                if (row, col) not in self.dug:
                    self.dig(row, col)
        return True
    
    def display_board(self):
        for y in range(len(self.board_list)):  # Iterate over rows
            for x in range(len(self.board_list[0])):  # Iterate over columns
                tile = self.board_list[x][y]
                if not tile.revealed:
                    print('F' if tile.flagged else '.', end=' ')
                elif tile.type == 'X':
                    print('X', end=' ')
                elif tile.number:
                    print(tile.number, end=' ')
                else:
                    print(' ', end=' ')
            print()



