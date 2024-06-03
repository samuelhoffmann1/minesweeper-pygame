import pygame
pygame.init() 
import pdb
from settings import *
from sprites import *


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()

    def new(self):
        self.board = Board()
        # self.board.display_board()
        self.playing = True
        self.win = False
        

    def run(self, tile):
        self.playing = True
        self.clock.tick(FPS)
        self.process_tile(tile)
        self.draw()
        if not self.playing or self.check_win():
            self.end_screen()

    def draw(self):
        self.screen.fill(BGCOLOUR)
        self.board.draw(self.screen)
        pygame.display.flip()

    def check_win(self):
        for row in self.board.board_list:
            for tile in row:
                if tile.type != "X" and not tile.revealed:
                    return False
        return True
    
    def process_tile(self, tile):
        if not tile.flagged:
            # dig and check if exploded
            if not self.board.dig(tile.x // TILESIZE, tile.y // TILESIZE):
                # explode
                for row in self.board.board_list:
                    for t in row:
                        if t.flagged and t.type != "X":
                            t.flagged = False
                            t.revealed = True
                            t.image = tile_not_mine
                        elif t.type == "X":
                            t.revealed = True
                self.playing = False

        if self.check_win():
            self.win = True
            self.playing = False
            for row in self.board.board_list:
                for t in row:
                    if not t.revealed:
                        t.flagged = True

    def end_screen(self):
       return




