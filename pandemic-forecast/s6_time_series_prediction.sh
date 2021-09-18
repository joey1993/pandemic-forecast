CUDA_VISIBLE_DEVICES=0 python experiments.py  \
--model GRU_GAT  \
--shift 7  \
--edge entity  \
--entity-entity \
--hiddens 64  \
--window 7  