- main: in questo file si utilizza il modello *dqn_maze_world.zip*, addestrato molto velocemente con pochi passi ma che riesce ad arrivare praticamente sempre a destinazione
---
- mazeWithSameHoles: in questo file si utilizza il modello *ppo_maze_world_with_same_holes.zip* addestrato con pochi step in un environment con le buche nella stessa posizione per tutto il training
---
- mazeWithDynamicHoles: in questo file si utilizzano i modelli *ppo_maze_world_with_dyn_holes_repstep_v2/v3.zip* addestrati con parecchi step in un environment in cui le buche cambiano ad ogni episodio, il v2 tutto in un unico ciclo, il v3 in un ciclo con incrementi degli step tra un model.save() e l'altro di 50k, riescono ad arrivare un discreto numero di volte a destinazione
---
- GuiExample: banale script che crea una griglia per un miglior effetto visivo durante il test del modello, utilizza i modelli *ppo_maze_world_with_dyn_holes_repstep_v2/v3.zip*
---
In *mazeWithDynamicHoles* e *GuiExample* si continuano a provare nuovi episodi, con le buche posizionate in modo diverso, fino a che l'episodio non termina con successo