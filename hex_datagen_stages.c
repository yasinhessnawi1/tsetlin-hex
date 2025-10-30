// Enhanced hex_datagen with multi-stage support
// Based on Ole-Christoffer Granmo's original hex_datagen.c
// Modification: Tracks board states at multiple stages (end, -2, -5)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#ifndef BOARD_DIM
    #define BOARD_DIM 11
#endif

#define MAX_STAGES 10

int neighbors[] = {-(BOARD_DIM+2) + 1, -(BOARD_DIM+2), -1, 1, (BOARD_DIM+2), (BOARD_DIM+2) - 1};

struct hex_game {
	int board[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
	int open_positions[BOARD_DIM*BOARD_DIM];
	int number_of_open_positions;
	int moves[BOARD_DIM*BOARD_DIM];
	int connected[(BOARD_DIM+2)*(BOARD_DIM+2)*2];

	// NEW: Store board states at multiple stages
	int num_moves;
	int board_history[BOARD_DIM*BOARD_DIM + 1][(BOARD_DIM+2)*(BOARD_DIM+2)*2];
};

void hg_init(struct hex_game *hg)
{
	for (int i = 0; i < BOARD_DIM+2; ++i) {
		for (int j = 0; j < BOARD_DIM+2; ++j) {
			hg->board[(i*(BOARD_DIM + 2) + j) * 2] = 0;
			hg->board[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;

			if (i > 0 && i < BOARD_DIM + 1 && j > 0 && j < BOARD_DIM + 1) {
				hg->open_positions[(i-1)*BOARD_DIM + j - 1] = i*(BOARD_DIM + 2) + j;
			}

			if (i == 0) {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 1;
			} else {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 0;
			}

			if (j == 0) {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 1;
			} else {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;
			}
		}
	}
	hg->number_of_open_positions = BOARD_DIM*BOARD_DIM;
	hg->num_moves = 0;
}

void hg_save_state(struct hex_game *hg)
{
	// Save current board state at index num_moves
	memcpy(hg->board_history[hg->num_moves], hg->board, sizeof(hg->board));
}

int hg_connect(struct hex_game *hg, int player, int position)
{
	hg->connected[position*2 + player] = 1;

	if (player == 0 && position / (BOARD_DIM + 2) == BOARD_DIM) {
		return 1;
	}

	if (player == 1 && position % (BOARD_DIM + 2) == BOARD_DIM) {
		return 1;
	}

	for (int i = 0; i < 6; ++i) {
		int neighbor = position + neighbors[i];
		if (hg->board[neighbor*2 + player] && !hg->connected[neighbor*2 + player]) {
			if (hg_connect(hg, player, neighbor)) {
				return 1;
			}
		}
	}
	return 0;
}

int hg_winner(struct hex_game *hg, int player, int position)
{
	for (int i = 0; i < 6; ++i) {
		int neighbor = position + neighbors[i];
		if (hg->connected[neighbor*2 + player]) {
			return hg_connect(hg, player, position);
		}
	}
	return 0;
}

int hg_place_piece_randomly(struct hex_game *hg, int player)
{
	if (hg->number_of_open_positions == 0) {
		return -1;
	}

	int pos = rand() % hg->number_of_open_positions;
	int position = hg->open_positions[pos];

	hg->open_positions[pos] = hg->open_positions[hg->number_of_open_positions - 1];
	hg->number_of_open_positions--;

	hg->board[position * 2 + player] = 1;

	return hg_winner(hg, player, position);
}

void print_board_state(int board[(BOARD_DIM+2)*(BOARD_DIM+2)*2])
{
	// Print board as CSV with proper encoding: 0=empty, 1=P0, 2=P1
	for (int i = 0; i < BOARD_DIM; ++i) {
		for (int j = 0; j < BOARD_DIM; ++j) {
			int pos = ((i+1)*(BOARD_DIM+2) + j + 1);
			if (board[pos*2] == 1) {
				printf("1");  // Player 0
			} else if (board[pos*2 + 1] == 1) {
				printf("2");  // Player 1
			} else {
				printf("0");  // Empty
			}
			if (i < BOARD_DIM-1 || j < BOARD_DIM-1) {
				printf(",");
			}
		}
	}
}

int main(int argc, char *argv[])
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <num_games> [stages...]\n", argv[0]);
		fprintf(stderr, "Example: %s 1000 0 -2 -5\n", argv[0]);
		fprintf(stderr, "  0 = end of game\n");
		fprintf(stderr, "  -2 = 2 moves before end\n");
		fprintf(stderr, "  -5 = 5 moves before end\n");
		return 1;
	}

	int num_games = atoi(argv[1]);

	// Parse stages (default: end only)
	int stages[MAX_STAGES];
	int num_stages = 0;

	if (argc > 2) {
		for (int i = 2; i < argc && num_stages < MAX_STAGES; i++) {
			stages[num_stages++] = atoi(argv[i]);
		}
	} else {
		// Default: end only
		stages[0] = 0;
		num_stages = 1;
	}

	srand(time(NULL));

	for (int game = 0; game < num_games; ++game) {
		struct hex_game hg;
		hg_init(&hg);

		int player = 0;
		int winner = -1;

		// Play game and record history
		while (hg.number_of_open_positions > 0 && winner == -1) {
			// Place piece and check if this player won
			int won = hg_place_piece_randomly(&hg, player);
			hg_save_state(&hg);
			hg.num_moves++;

			if (won) {
				winner = player;  // Set winner to the player who won!
				break;
			}

			player = 1 - player;
		}

		// Output: winner,stage0_data,stage1_data,...
		if (winner != -1) {
			printf("%d", winner);

			for (int s = 0; s < num_stages; s++) {
				int stage = stages[s];
				int move_index;

				if (stage > 0) {
					// Absolute position from start (after move N)
					move_index = stage;
				} else if (stage == 0) {
					// Stage 0 means END (final move)
					move_index = hg.num_moves - 1;
				} else {
					// Relative to end: stage -1 = 1 before end, -2 = 2 before end
					// Formula: (end_index) + stage = (num_moves - 1) + (-2) = num_moves - 3
					move_index = (hg.num_moves - 1) + stage;
				}

				// Clamp to valid range [0, num_moves-1]
				if (move_index < 0) move_index = 0;
				if (move_index >= hg.num_moves) move_index = hg.num_moves - 1;

				printf(",");
				print_board_state(hg.board_history[move_index]);
			}

			printf("\n");
		}

		if ((game + 1) % 1000 == 0) {
			fprintf(stderr, "Generated %d games...\n", game + 1);
		}
	}

	fprintf(stderr, "Done! Generated %d games.\n", num_games);
	return 0;
}
