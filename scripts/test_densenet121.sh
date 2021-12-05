CUDA_VISIBLE_DEVICES=3 python3 test.py \
--vision-backbone densenet121 \
--textual-embeddings embeddings/nih_chest_xray_biobert.npy \
--load-from working_copies/checkpoints-t,t--pre-new/9/best_auroc_checkpoint.pth.tar \
--vae-load-from working_copies/checkpoints-t,t--pre-new/9/best_auroc_vae-backprop.pth.tar
