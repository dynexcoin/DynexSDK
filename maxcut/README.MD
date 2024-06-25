# Update: Dynex finds new best-known MaxCut for the Stanford G70 benchmark

The better cut size of [9,555](https://github.com/dynexcoin/DynexSDK/blob/main/maxcut/maxcut_best_9555.pkl) for [G70](https://web.stanford.edu/~yyye/yyye/Gset/) found by [Dynex](https://github.com/dynexcoin/DynexSDK/blob/main/maxcut/G70_dynex.ipynb) was found on June 24th, 2024 on the Dynex n.quantum computing cloud.

# Update: Dynex finds best-known MaxCut for the Stanford G70 benchmark [G70](https://web.stanford.edu/~yyye/yyye/Gset/) found by [Dynex](https://github.com/dynexcoin/DynexSDK/blob/main/maxcut/G70_dynex.ipynb) was found on June 24th, 2024.

The cut size [9,553](https://github.com/dynexcoin/DynexSDK/blob/main/maxcut/maxcut_best_9553.pkl) for [G70](https://web.stanford.edu/~yyye/yyye/Gset/) found by [Dynex](https://github.com/dynexcoin/DynexSDK/blob/main/maxcut/G70_dynex.ipynb) is the best-known MaxCut than previous reported results on June 8th 2024. The previous record of 9,541 was found using a special purpose algorithm designed for MaxCut ([Breakout Local Search for the Max-Cut problem](https://www.researchgate.net/publication/257392755_Breakout_Local_Search_for_the_Max-Cutproblem)) in 2013. It is worth noting that an increase of one better cut involves an exponentially increasing complexity. 

## Validating the MaxCut solution of 9,553

The solutions in this repository can be easily validated with the provided [Jupyter Notebook](https://github.com/dynexcoin/DynexSDK/blob/main/maxcut/g70solutionvalidator.ipynb) or by using the following lines of code:

```
import networkx as nx
import pickle

filename = 'G70.dat' # obtained from Stanford website: https://web.stanford.edu/~yyye/yyye/Gset/
solution = 'maxcut_best_9553.pkl'

# load g70 problem
G = nx.Graph();
f = open(filename, "r").readlines()
for ele in f[1:]:
    i, j, v = ele.split()
    G.add_edges_from([(int(i),int(j))])

# load solution
with open(solution,'rb') as inp:
    lut = pickle.load(inp)

# Validate cut result:
S0 = [node for node in G.nodes if not lut['X['+str(node)+']']]
S1 = [node for node in G.nodes if lut['X['+str(node)+']']]
cut_edges = [(u, v) for u, v in G.edges if lut['X['+str(u)+']'] != lut['X['+str(v)+']']]
uncut_edges = [(u, v) for u, v in G.edges if lut['X['+str(u)+']'] == lut['X['+str(v)+']']]

print('Maxcut result:',len(cut_edges))

>> Maxcut result: 9553
```

## Previously known best MaxCuts

The following table displays the previously known best MaxCut values for various Gset benchmarks, including the G70 dataset. Results for D-Wave have been computed using the D-Wave Cloud Platform using the Advantage™ quantum computer with 5,000 physical qubits. Improving the cut size by even one involves exponentially increasing complexity.

| graph | #nodes| #edges | DYNEX | BLS | DWAVE | DSDP    | KHLWG   | RUN-CSP| PI-GNN| Gurobi (0.5 h)  | Gurobi (1 h)  | 
|--- |------|----  |---  |------ |------ |-----    |-----    |--------|-------|------           | ---           |
|G14 | 800  | 4694 |        | 3064      |       |         | 2922    |  3061  | 2943  | 3034            |3042           |
|G15 | 800  | 4661 |        | 3050      |       |  2938   | 3050    |  2928  | 2990  | 3016            | 3033          | 
|G22 | 2000 | 19990|        |13359      |       | 12960   |13359    | 13028  | 13181 |13062            |13129          | 
|G49 | 3000 | 6000 |        | 6000      |       |  6000   | 6000    |  6000  | 5918  | 6000            |  6000         | 
|G50 | 3000 | 6000 |        | 5880      |       |  5880   | 5880    |  5880  | 5820  | 5880            |  5880         | 
|G55 | 5000 | 12468|        |10294      |       |  9960   |10236    | 10116  | 10138 |10103            | 10103         | 
|G70 | 10000| 9999 |__9553__| 9541      | 9546  |  9456   | 9458    |        | 9421  | 9489            | 9490          | 

# Update: Dynex Enhanced Core Algorithm

The Maximum Cut value was further improved to [9,543](https://github.com/dynexcoin/DynexSDK/blob/main/maxcut/maxcut_best_9543.pkl) on May 26th 2024 using the enhanced core algorithm.

# Enhancing MaxCut Solutions: Dynex’s Benchmark Performance on G70 Using Quantum Computing

We conducted an experiment on the G70 MaxCut problem, a complex unweighted MaxCut challenge with 10,000 nodes and 99,999 edges sourced from the publicly available Gset dataset provided by Stanford University. Various algorithms and solvers were evaluated, including commercial solvers like Gurobi and IBM CPLEX, and specialized algorithms such as the SDP solver DSDP, the graph neural network-based PI-GNN, and the tabu search-based KHLWG. Utilizing Dynex’s neuromorphic quantum computing cloud with a single Nvidia RTX3080 GPU, and without any MaxCut-specific enhancements, Dynex achieved a Maximum Cut value of 9,506 — outperforming other algorithms and demonstrating the potential of the Dynex platform. The experiment underscores that additional computational resources or bespoke algorithmic modifications could potentially lead to even higher performance, potentially setting new benchmarks for MaxCut solutions and showcasing the scalability and adaptability of the Dynex platform for tackling such optimization problems.

- [Jupyter Noteboook](https://github.com/dynexcoin/DynexSDK/blob/main/maxcut/G70_dynex.ipynb)

# Dynex Sets New World Record for Quantum Computing, Breaking NVIDIA's Previous Record

We are pleased to announce a significant breakthrough in the field of quantum computing, achieved through the Dynex neuromorphic quantum computing platform. By employing a sophisticated quantum algorithm, we successfully solved a graph containing 10,000 vertices, setting a new record that surpasses the previous benchmark by threefold. The prior record was established by NVIDIA using their cuQuantum software on the in-house supercomputer, Selene. NVIDIA's effort involved 896 GPUs to simulate 1,688 qubits, capable of resolving a graph with 3,375 vertices. This achievement, at the time, represented an eightfold increase in the number of qubits compared to the largest previous quantum simulation. Our recent accomplishment not only advances the capabilities of quantum algorithms but also significantly expands the potential applications of quantum computing in solving complex optimization problems.

The computation involved only a small fraction of the Dynex’ network, using only 19 workers for around 15 minutes from a total of more than 120,000 available GPUs. The computation involved 18,653 quantum gates and 10,009 Qubits.

- [Jupyter Notebook](https://github.com/dynexcoin/DynexSDK/blob/main/maxcut/maxcut_record.ipynb)
