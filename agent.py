import pygame
from settings import *
from sprites import *
from main import Game
from MineGroup import *
import matplotlib.pyplot as plt
import numpy as np

class MinesweeperAgent:
    def __init__(self, game):
        self.game = game
        self.board = game.board
        self.remaining_mines = AMOUNT_MINES
        self.covered_tiles = ROWS * COLS
        self.groups = AllGroups()
        self.all_clusters = AllClusters(self.covered_tiles,
                                           self.remaining_mines, self.game)
        self.probability = None
        self.unaccounted_group = None
        self.bruteforce_solutions = None
        self.first_move = True
        self.last_move_info = None
        self.random_selections = 0  # Counter for random tile selections

    def generate_all_covered(self):
        ''' Populate self.covered_tiles with the list of all covered cells.
        '''
        self.covered_tiles = self.board.get_covered_tiles()

    def generate_remaining_mines(self):
        ''' Populate self.mines with the list of all mines.
        '''
        self.remaining_mines = self.board.calculate_remaining_mines()

    def generate_groups(self):
        ''' Populate self.group with MineGroup objects
        '''

        # Reset the groups
        self.groups.reset()

        # Go over all cells and find all the "Numbered ones"
        for row in self.board.board_list:
            for tile in row:

                # Groups are only for numbered cells
                if not tile.number or not tile.revealed:
                    continue

                # For them we'll need to know two things:
                # What are the uncovered cells around it
                covered_neighbors = []
                # And how many "Active" (that is, minus marked)
                # mines are still there
                active_mines = tile.number

                # Go through the neighbors
                for neighbor in self.board.get_neighbours(tile.x // TILESIZE, tile.y // TILESIZE):
                    # Collect all covered cells
                    if not neighbor.revealed and not neighbor.flagged:
                        covered_neighbors.append(neighbor)
                    # Subtract all marked mines
                    if neighbor.flagged:
                        active_mines -= 1

                # If the list of covered cells is not empty:
                # store it in the self.groups
                if covered_neighbors:
                    new_group = MineGroup(covered_neighbors, active_mines)
                    self.groups.add_group(new_group)

    def generate_clusters(self):
        ''' Initiate self.all_clusters and populate it with
        GroupCluster objects
        '''
        # Reset clusters
        self.all_clusters = AllClusters(self.covered_tiles,
                                           self.remaining_mines, self.game)
        # Reset all "belong to cluster" information from the groups
        self.groups.reset_clusters()

        # Keep going as long as there are groups not belonging to clusters
        while self.groups.next_non_clustered_groups() is not None:

            # Initiate a new cluster with such a group
            new_cluster = GroupCluster(
                self.groups.next_non_clustered_groups())

            while True:

                # Look through all groups
                for group in self.groups.exact_groups():
                    # If it overlaps with this group and not part of any
                    # other cluster - add this group
                    if group.belong_to_cluster is None and \
                       new_cluster.overlap(group):
                        new_cluster.add_group(group)
                        break

                # We went through the groups without adding any:
                # new_cluster is done
                else:
                    self.all_clusters.clusters.append(new_cluster)
                    # Exit the while loop
                    break

    def generate_unaccounted(self):
        ''' Populate self.unaccounted_group with a MineGroup made of cells
        and mines from "unknown" area, that is NOT next to any number.
        Used in "Coverage" method and in mine probability calculations.
        '''

        def coverage_attempt(accounted_cells, accounted_mines):
            ''' Create a coverage (set of non-overlapping mine groups),
            given cells and mines that are already in the coverage.
            Uses greedy method to maximize the number of cells in the coverage
            '''

            while True:
                # The idea is to find a group that has a largest number of
                # unaccounted cells
                best_count = None
                best_group = None
                for group in self.groups.exact_groups():

                    # If group overlaps with what we have so far -
                    # we don't need such group
                    if accounted_cells.intersection(group.cells):
                        continue

                    # Find the biggest group that we haven't touched yet
                    if best_count is None or len(group.cells) > best_count:
                        best_count = len(group.cells)
                        best_group = group

                # We have a matching group
                if best_group is not None:
                    # Cells from that group from now on are accounted for
                    accounted_cells = accounted_cells.union(best_group.cells)
                    # And so are  mines
                    accounted_mines += best_group.mines
                # No such  group was found: coverage is done
                else:
                    break

            return accounted_cells, accounted_mines

        # Reset the final variable
        self.unaccounted_group = None

        # This method usually has no effect in the beginning of the game.
        # Highest count of remaining cells when it worked was at 36-37.
        if len(self.covered_tiles) > 40:
            return

        # Generate several coverage options.
        # Put them into coverage_options
        coverage_options = []
        # FOr each option we are going to start with different group,
        # and then proceed with greedy algorithm
        for group in self.groups:
            initial_cells = set(group.cells)
            initial_mines = group.mines
            coverage_option_cells, coverage_option_mines = \
                coverage_attempt(initial_cells, initial_mines)
            coverage_options.append((coverage_option_cells,
                                     coverage_option_mines))

        if not coverage_options:
            return

        # Sort them by the number of cells in coverage
        # Choose the one with the most cells
        coverage_options.sort(key=lambda x: len(x[0]), reverse=True)
        accounted_cells, accounted_mines = coverage_options[0]

        # unaccounted cells are all minus accounted
        unaccounted_cells = set(self.covered_tiles).difference(accounted_cells)
        # Same with mines
        unaccounted_mines = self.remaining_mines - accounted_mines

        # Those unaccounted mines can now for a new  group
        self.unaccounted_group = MineGroup(unaccounted_cells,
                                              unaccounted_mines)

    def method_groups(self):
        ''' Method #2. Groups.
        Cross check all groups. When group is a subset of
        another group, try to deduce safe cells and mines.
        '''
        safe, mines = [], []

        # Cross-check all-with-all groups
        for group_a in self.groups:
            for group_b in self.groups:

                # Don't compare with itself
                if group_a.hash == group_b.hash:
                    continue

                safe.extend(self.deduce_safe(group_a, group_b))
                mines.extend(self.deduce_mines(group_a, group_b))

                # If group A is a subset of group B and B has more mines
                # we can create a new group that would contain
                # B-A cells and B-A mines
                # len(group_b.cells) < 8 prevents computational explosion on
                # multidimensional fields
                if len(group_b.cells) < 8 and \
                   group_a.cells.issubset(group_b.cells) and \
                   group_b.mines - group_a.mines > 0:
                    new_group = MineGroup(group_b.cells - group_a.cells,
                                             group_b.mines - group_a.mines)
                    self.groups.add_group(new_group)

        return list(set(safe))
    
    def method_csp(self):
        ''' Method #4. CSP (Constraint Satisfaction Problem).
        Generate overlapping groups (clusters). For each cluster find safe
        cells and mines by brute forcing all possible solutions.
        '''
        safe, mines = [], []

        # Generate clusters
        self.generate_clusters()
        # Do all the solving / calculate frequencies stuff
        self.all_clusters.calculate_all()

        for cluster in self.all_clusters.clusters:

            # Get safe cells and mines from cluster
            safe.extend(cluster.safe_cells())
            mines.extend(cluster.mine_cells())
        return list(set(safe))

    def calculate_probabilities(self):
        ''' Final method. "Probability". Use various methods to determine
        which cell(s) is least likely to have a mine
        '''

        def background_probabilities(self):
            ''' Populate self.probabilities based on background probability.
            Which is All mines divided by all covered cells.
            It is quite crude and often inaccurate, it is just a fallback
            if any of more accurate methods don't work.
            '''
            background_probability = \
                self.remaining_mines / len(self.covered_tiles)

            for cell in self.covered_tiles:
                self.probability.cells[cell] = \
                    CellProbability(cell, "Background",
                                       background_probability)

        def probabilities_for_groups(self):
            ''' Update self.probabilities, based on mine groups.
            For each group consider mine probability as "number of mines
            divided by the number of cells".
            '''
            for group in self.groups.exact_groups():

                # Probability of each mine in the group
                group_probability = group.mines / len(group.cells)
                for cell in group.cells:

                    # If group's probability is higher than the background:
                    # Overwrite the probability result
                    if group_probability > \
                       self.probability.cells[cell].mine_chance:
                        self.probability.cells[cell] = \
                            CellProbability(cell, "Groups",
                                               group_probability)

        def csp_probabilities(self):
            ''' Update self.probabilities based on results from CSP method.
            '''
            for cluster in self.all_clusters.clusters:
                for cell, frequency in cluster.frequencies.items():
                    # Overwrite the probability result
                    self.probability.cells[cell] = \
                        CellProbability(cell, "CSP", frequency)

        # Reset probabilities
        self.probability = AllProbabilities()
        # Background probability: all remaining mines on all covered cells
        background_probabilities(self)
        # Based on mines in groups
        probabilities_for_groups(self)
        # Based on CSP solutions
        csp_probabilities(self)

    def calculate_opening_chances(self):
        ''' Populate opening_chance in self.probabilities: a chance that this
        cell is a zero. (Which is a good thing)
        '''
        # Go through all cells we have probability info for
        # (that would be all covered cells)
        for cell, cell_info in self.probability.cells.items():
            zero_chance = 1
            # Look at neighbors of each cell
            for neighbor in self.board.get_neighbours(cell.x // TILESIZE, cell.y // TILESIZE):
                # If there are any mines around, there is no chance of opening
                if neighbor.flagged:
                    cell_info.opening_chance = 0
                    break
                # Otherwise each mine chance decrease opening chance
                # by (1 - mine chance) times
                if neighbor in self.probability.cells:
                    zero_chance *= \
                        (1 - self.probability.cells[neighbor].mine_chance)
            else:
                self.probability.cells[cell].opening_chance = zero_chance

    def calculate_frontier(self):
        ''' Populate frontier (how many groups may be affected by this cell)
        '''
        # Generate frontier
        self.groups.generate_frontier()

        for cell in self.groups.frontier:
            for neighbors in self.board.get_neighbours(cell.x // TILESIZE, cell.y // TILESIZE):
                if neighbors in self.probability.cells:
                    self.probability.cells[neighbors].frontier += 1

    def calculate_next_safe_csp(self):
        ''' Populate "next safe" information (how many guaranteed safe cells
        will be in the next move, based on CSP solutions).
        '''
        # Do the calculations
        self.all_clusters.calculate_all_next_safe()

        # Populate probability object with this info
        for cluster in self.all_clusters.clusters:
            for cell, next_safe in cluster.next_safe.items():
                self.probability.cells[cell].csp_next_safe = next_safe

    def make_decision(self):
        safe_tiles = []
        # self.board.display_board()
        if self.first_move:
            self.first_move = False
            return [self.board.board_list[0][0]]
        else:
            self.find_mines()
            safe_tiles = self.basic()
            if safe_tiles:
                self.last_move_info = ("Naive", None, None)
            if not safe_tiles:
                self.last_move_info = ("CSP", None, None)
                safe_tiles = self.method_csp()
            if not safe_tiles:
                # Calculate mine probability using various methods
                self.calculate_probabilities()
                # Calculate safe cells for the next move in CSP
                self.calculate_next_safe_csp()

                # Two more calculations that will be used to pick
                # the best random cell:
                # Opening chances (chance that cell is a zero)
                self.calculate_opening_chances()
                # Does it touch a frontier (cells that already are in groups)
                self.calculate_frontier()

                lucky_cells = self.probability.get_luckiest()
                if lucky_cells:
                    self.random_selections += 1
                    return [lucky_cells[0]]
                self.last_move_info = ("Last Resort", None, None)
                return [self.board.get_random_unrevealed_tile()]
            return safe_tiles


    def play(self):
        self.generate_all_covered()
        self.generate_remaining_mines()
        self.generate_groups()
        self.generate_unaccounted()
        tiles = self.make_decision()
        for tile in tiles:
            self.game.draw()  # Draw the board
            self.game.run(tile)
    
    def find_mines(self):
        for row in range(ROWS):
            for col in range(COLS):
                tile = self.board.board_list[row][col]
                unrevealed_neighbors = [neighbour for neighbour in self.board.get_neighbours(row, col) if not neighbour.revealed]
                if tile.number == len(unrevealed_neighbors):
                    for mine in unrevealed_neighbors:
                        mine.flagged = True

    def basic(self):
        for row in range(ROWS):
            for col in range(COLS):
                tile = self.board.board_list[row][col]
                if tile.number:
                    unrevealed_neighbors = [neighbour for neighbour in self.board.get_neighbours(row, col) if not neighbour.revealed]
                    flagged_neighbors = [neighbour for neighbour in unrevealed_neighbors if neighbour.flagged]
                    if tile.number == len(flagged_neighbors) and len(unrevealed_neighbors) > len(flagged_neighbors):
                        return [neighbour for neighbour in unrevealed_neighbors if not neighbour.flagged]

# Main game loop with the agent
if __name__ == "__main__":
    pygame.init()
    random.seed(14)
    wins = 0  # Counter for wins
    losses = 0  # Counter for losses
    total_random_selections = 0  # Counter for total random tile selections

    for i in range(100):  # Play 100 games
        game = Game()
        game.new()
        agent = MinesweeperAgent(game)
        while game.playing:
            agent.play()
            if not game.playing:
                break

        if game.win:  # If the game was won, increment the wins counter
            wins += 1
        else:  # If the game was lost, increment the losses counter
            losses += 1

        total_random_selections += agent.random_selections  # Add the number of random tile selections in this game to the total

    win_percentage = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0  # Calculate the win percentage, avoid division by zero
    average_random_selections = total_random_selections / 100  # Calculate the average number of random choices per game

    print("{:<20} {:<10} {:<10} {:<20} {:<25}".format('Wins', 'Losses', 'Win %', 'Total Random Selections', 'Avg Random Selections'))  # Print the table headers
    print("{:<20} {:<10} {:<10.2f} {:<20} {:<25.2f}".format(wins, losses, win_percentage, total_random_selections, average_random_selections))  # Print the table data

    pygame.quit()