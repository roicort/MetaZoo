import numpy as np

class MelodyOperators:
    @staticmethod
    def crossover(parent1, parent2):
        # Dos puntos de corte para mayor variedad
        p1, p2 = sorted(np.random.choice(range(1, len(parent1)), 2, replace=False))
        child1 = np.vstack((parent1[:p1], parent2[p1:p2], parent1[p2:]))
        child2 = np.vstack((parent2[:p1], parent1[p1:p2], parent2[p2:]))
        return child1, child2

    @staticmethod
    def mutation(individual, note_range=(60,72), durations=(0.25,0.5,1.0,2.0), silence_prob=0.05):
        idx = np.random.randint(len(individual))
        # Evita mutar a silencio si ya hay muchos
        if np.random.rand() < silence_prob and sum(individual[:,0]==0)/len(individual) < 0.3:
            individual[idx, 0] = 0
        else:
            individual[idx, 0] = np.random.randint(note_range[0], note_range[1])
        individual[idx, 1] = np.random.choice(durations)
        return individual