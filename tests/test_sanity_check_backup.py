import numpy as np
d = np.load('data/train_games_5x5.npz')
print('First 5 games:')
for i in range(5):
    print(f'\nGame {i}: Winner={d["winners"][i]}')
    b = d['states_at_end'][i]
    print(b)
    print(f'P0={np.sum(b==1)}, P1={np.sum(b==2)}, Empty={np.sum(b==0)}')

