"""Population initialization strategies for the FedGP genetic-programming engine."""

def init_elite_random(elites, population_size, toolbox):
    """Create a population from the carried-over elites plus random individuals.

    ``elites`` is the list of best individuals carried from the previous round (its length
    is the configured ``elitism_size``). They are cloned in front, and the remaining slots
    are filled with random individuals.
    """
    population = [toolbox.clone(elite) for elite in elites]
    n_random = max(0, population_size - len(population))
    population += toolbox.population(n=n_random)
    return population[:population_size]


def init_elite_mutation(elites, population_size, toolbox):
    """Create a population from the carried-over elites and mutations of them.

    The elites are cloned in front; the remaining slots are filled by mutating clones of the
    elites, cycling through them so the mutated offspring stay diverse rather than all
    deriving from the single best.
    """
    population = [toolbox.clone(elite) for elite in elites]
    n_mut = max(0, population_size - len(population))
    for i in range(n_mut):
        ind = toolbox.clone(elites[i % len(elites)])
        ind, = toolbox.mutate(ind)
        population.append(ind)
    return population[:population_size]


def init_hybrid_population(elites, population_size, toolbox, alpha=0.5):
    """Create a mixed population: carried-over elites, their mutations, and random individuals.

    The elites are cloned in front; the remaining slots are split between mutations of the
    elites (a fraction ``alpha``, cycling through them) and fresh random individuals.
    """
    population = [toolbox.clone(elite) for elite in elites]
    remaining = max(0, population_size - len(population))

    n_mut = int(alpha * remaining)
    n_rand = remaining - n_mut

    for i in range(n_mut):
        ind = toolbox.clone(elites[i % len(elites)])
        ind, = toolbox.mutate(ind)
        population.append(ind)

    population += toolbox.population(n=n_rand)
    return population[:population_size]
