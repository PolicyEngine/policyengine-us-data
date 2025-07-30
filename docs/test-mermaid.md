# Test Mermaid Version

Testing horizontal layout with direction LR in subgraphs:

```{mermaid}
flowchart TD
    subgraph src["Source Datasets"]
        direction LR
        CPS["CPS ASEC"]
        PUF["IRS PUF"]
        SIPP["SIPP"]
        SCF["SCF"]
        ACS["ACS"]
    end
```

Testing with %%init%% directive:

```{mermaid}
%%{init: {'theme':'base'}}%%
flowchart LR
    subgraph datasets["Datasets"]
        A[Dataset A] 
        B[Dataset B]
        C[Dataset C]
    end
```

Testing simple horizontal flow:

```{mermaid}
flowchart LR
    A[Box A] --> B[Box B] --> C[Box C]
```