# trivago-recsys-moe

## Summary of Components
* **ContextAwareSASRec**: A 103M-parameter transformer trained on 649k Trivago sessions.
* **MoE Bridge**: Persona-conditioned routing over 7 archetypes from 50k Nvidia Nemotron embeddings.
* **Sequential Pre-Training**: Pre-trained on 9.2M Yoochoose retail sessions.
