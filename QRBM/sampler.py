def sample_opposite_layer_minorminer(v, layer, weights, opposite_layer: int):
    raise NotImplementedError

from pyqubo import Binary
import dynex

def sample_opposite_layer_pyqubo(v, layer, weights, opposite_layer,
                                 qpu=False, chain_strength=2, num_reads=1):
    # initialize Hamiltonian
    H = 0
    H_vars = []

    # print('opposite_layer:',opposite_layer)

    # initialize all variables (one for each node in the opposite layer)
    for j in range(len(opposite_layer)):
        H_vars.append(Binary('DNX'+str(j)))

    for i, bias in enumerate(layer):
        # filter only chosen nodes in the first layer
        if not v[i]:
            continue

        # add reward to every connection
        for j, opp_bias in enumerate(opposite_layer):
            H += -1 * weights[i][j] * H_vars[j]

    for j, opp_bias in enumerate(opposite_layer):
        H += -1 * opp_bias * H_vars[j]


    model = H.compile()
    # print(model)
    bqm = model.to_bqm()
    
    # use the dynex sampler:
    dnxmodel = dynex.BQM(bqm, logging=False);
    dnxsampler = dynex.DynexSampler(dnxmodel, logging=False, mainnet=False, description='QRBM');
    sampleset = dnxsampler.sample(num_reads=4096, annealing_time = 1000);
    solution1 = sampleset.first.sample;

    solution1_list = [(k[3:], v) for k, v in solution1.items()]
    solution1_list.sort(key=lambda tup: int(tup[0]))  # sorts in place
    solution1_list_final = [v for (k, v) in solution1_list]
    return solution1_list_final


def sample_v(layer, qpu=False, chain_strength=10, num_reads=1):
    H = 0

    for i, bias in enumerate(layer):
        H += -1 * bias * Binary('DNX'+str(i))

    model = H.compile()
    bqm = model.to_bqm()

    # use the dynex sampler:
    dnxmodel = dynex.BQM(bqm, logging=False);
    dnxsampler = dynex.DynexSampler(dnxmodel, logging=False, mainnet=False);
    sampleset = dnxsampler.sample(num_reads=4096, annealing_time = 1000);
    solution1 = sampleset.first.sample;

    solution1_list = [(k, v) for k, v in solution1.items()]
    solution1_list.sort(key=lambda tup: int(tup[0]))  # sorts in place
    solution1_list_final = [v for (k, v) in solution1_list]
    return solution1_list_final

if __name__ == "__main__":

    # example from https://www.youtube.com/watch?v=Fkw0_aAtwIw (22:50)
    v = [0,1,0]
    layer = [1,1,1]
    weights = [[2,-2], [-4, 4], [2,-2]]
    opp_layer = [2,1]

    sampleset = sample_opposite_layer_pyqubo(v, layer, weights, opp_layer,
                                             qpu=True)
    sampleset2 = sample_v(layer, qpu=True)
    print(sampleset)
    print(sampleset2)

    # using the best (lowest energy) sample
    solution1 = sampleset.first.sample
    print(solution1)
    solution1_list = [(k, v) for k, v in solution1.items()]
    solution1_list.sort(key=lambda tup: int(tup[0]))  # sorts in place
    solution1_list_final = [v for (k, v) in solution1_list]
