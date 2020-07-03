# Comparison between Neo4j and Networkx
## Introduction

This notebook aims to compare some algorithms implemented in both Neo4j Graph Data Science and Python's package Networkx.
I used the database from https://github.com/mathbeveridge/asoiaf corresponding to interactions between the characters of the books of the saga Song of Ice and Fire by G.R.R Martin.

I ran all the tests on my PC with the following configuration :
- Intel Core  i5-9300H 2,40 GHz, 4 core, 8 Mo cache memory (turbo 4,10 GHz)
- 12 GB RAM 

In order to compare networkx and Neo4j, I chose to compare the different implemantion of Betweenness centrality, Page Rank, Label Propagation, BFS and MST.
They are different type of algorithms which are implemented both in Neo4j and Networkx. Much of the algorithms are implemented in only one of them.
I will compare the average time to compute these algorithms with the same graph. 

Due to the implementation of Neo4j's graphs, I had considered 2 different mesures of time. One is the first iteration of the algorithm and the other is the average time of the following iterations where the Neo4j's cache memory reduces the computation time.


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
```

## Creation of the GOT's Graph


```python
table=pd.read_csv('./data/data/asoiaf-all-edges.csv')
```


```python
import networkx as nx
G=nx.Graph(name="Game of Networks")
n=len(table['Source'])
for i in range(n):
    G.add_edge(table['Source'][i],table['Target'][i],weight=table['weight'][i])


```

## Networkx's tests

### Betweenness Centrality




```python
from time import *
```


```python
liste=[]
for i in range(100):
    a=time()
    nx.betweenness_centrality(G)
    b=time()
    liste.append(b-a)
print(sum(liste)/100)
```

    1.6309340572357178


### PageRank


```python
liste=[]
for i in range(100):
    a=time()
    nx.pagerank(G,alpha=0.85,max_iter=20)
    b=time()
    liste.append(b-a)
print(sum(liste)/100)

```

    0.12770500183105468


### Label Propagation


```python

liste=[]
for i in range(100):
    a=time()
    c=nx.algorithms.community.label_propagation.label_propagation_communities(G)
    b=time() 
    liste.append(b-a)
print(sum(liste)/100)

```

    1.3613700866699219e-06



### Minimum Spanning Tree



```python

liste=[]
for i in range(100):
    a=time()
    nx.minimum_spanning_tree(G)
    b=time() 
    liste.append(b-a)
print(sum(liste)/100)
 
```

    0.007812273502349853


### BFS


```python
liste=[]
for i in range(100):
    a=time()
    t= nx.bfs_edges(G,"Jon-Snow",depth_limit=5)
    b=time() 
    liste.append(b-a)
print(sum(liste)/100)
print(liste[:5])
```

    6.320476531982422e-06
    [1.0728836059570312e-05, 5.4836273193359375e-06, 1.9073486328125e-06, 1.430511474609375e-06, 1.430511474609375e-06]


## Tests Neo4j (ms)
This test were not really automated because I wanted to pass through the cache memory in order to have to really time of algorithms. Therefore I had to restart the server between each test. That is why there not has much data as networkx. However, I think this is enough to have a decent overview of the capacity of each one.

### Code used:
#### Graph in memory
```
CALL gds.graph.create.cypher(
    'G',
    'MATCH (n) RETURN id(n) AS id',
    'MATCH (a)-[]-(b) RETURN id(a) AS source, id(b) AS target'
)
YIELD graphName, nodeCount, relationshipCount, createMillis;
```

#### Betweenness centrality
```
 CALL gds.alpha.betweenness.stream({
nodeQuery: 'MATCH (p) RETURN id(p) AS id',
  relationshipQuery: 'MATCH (p1)-[]-(p2) RETURN id(p1) AS source, id(p2) AS target'
})
YIELD nodeId,centrality
return gds.util.asNode(nodeId).name as user,centrality
order by centrality DESC limit 1
```
#### Page rank
```
CALL gds.pageRank.stream('G',{maxIterations:20, dampingFactor:0.85})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC, name ASC limit 1
```

####  Label Propagation
```
CALL gds.labelPropagation.stream('G'})
YIELD nodeId, communityId AS Community
RETURN gds.util.asNode(nodeId).name AS Name, Community
ORDER BY Community, Name limit 1 
```

#### BFS
```
MATCH (a:Character{name:'Jon Snow'})
WITH id(a) AS startNode
CALL gds.alpha.bfs.stream('G', {startNode: startNode,maxDepth :5})
YIELD path
UNWIND [ n in nodes(path) | n.name] AS name
RETURN name
ORDER BY name limit 1
```

#### MST
```
MATCH (n:Character {name: 'Jon Snow'})
CALL gds.alpha.spanningTree.minimum.write({
  nodeProjection: 'Character',
  relationshipProjection: {
  	Interacts: {
    	type :'Interacts',
   	 	properties: 'weight',
    	orientation: 'UNDIRECTED'
    	}
    },
    startNodeId: id(n),
    relationshipWeightProperty: 'weight',
    writeProperty: 'MINST',
    weightWriteProperty: 'writeCost'
})
YIELD createMillis, computeMillis, writeMillis, effectiveNodeCount
RETURN createMillis, computeMillis, writeMillis, effectiveNodeCount;
```

### Raw results: 
betweenness #1 (ms) : 354,90,67,76,59,85,55,61,84,58,69

page rank #1 (ms) : 327,202,166, crash

betweenness #2 (ms) : 288,95,85,61,65,69,68,69

page rank #2 (ms) : 294,191,146

page rank #3 (ms) : 451, 241, 154,

label propagation #1 (ms) : 230, 89,88,68,47,62,47,43,47

label propagation #2 (ms) : 120, 62,51,33,27,38,30

label propagation #3 (ms) : 170, 74, 64,79,44,49,47,46,67,

BFS (graph in memory) : environ 37;38,7;38,5 seconds for  max depth=5

MST :454,42 

MST(with graph projection during the algo) : , 1100, 1255,

MST (others first iterations) :488,483, 583,

MST:  620,66,57,53,60,46

## Synthesis
Algorithm | Time Neo4j (s) | Time Networkx(s)
:-: | :-: | :-:
Betweenness centrality | 0,32 (0,08 after) | 1,6
Page rank | 0,36 (0,1 after) | 0,12 
Label Propagation | 0,17 (0,05 after) | 1e-6  (1e-5 for the first one)
Breadth First Search | 38 | 3e-7 (2e-6 for the 2 first ones)
Minimum spanning tree | 0,52 (0,06 after)| 7e-3 




## Conclusion

It is hard to have a clear decision because there is huge gap for either software. Most of the time Networkx is far better than neo4j. But is some case (Betweenness and Page rank) Neo4j is better and getting even better after the first iteration due to the use of a cache. 
However, there is a huge gap between Neo4j and Network regarding to BFS's algorithm (1e8 order of magnitude...)

