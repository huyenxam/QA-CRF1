#!/bin/bash
export TRAIN_PATH="DataNew/train_ViQuAD.json"
export DEV_PATH="DataNew/dev_ViQuAD.json"
export TEST_PATH="DataNew/test_ViQuAD.json"
export CHAR_VOCAB_PATH="data/charindex.json"
export LABEL_SET_PATH="data/label_set.txt"
export MAX_CHAR_LEN=20
export MAX_SEQ_LENGTH=255
export BATCH_SIZE=2
export CHAR_EMBEDDING_DIM=100
export CHAR_HIDDEN_DIM=200
export NUM_BERT_LAYER=1
export CHAR_VOCAB_SIZE=108
export HIDDEN_DROPOUT_PROB=0.1 
export HIDDEN_DIM=728
export NUM_LABELS=7
export MODEL_NAME_OR_PATH="vinai/phobert-base"
export NUM_EPOCHS=20
export LEARNING_RATE=5e-6
export ADAM_EPSILON=1e-8
export WEIGHT_DECAY=0.01
export WARMUP_STEPS=0
export MAX_GRAD_NORM=1
export SAVE_FOLDER="results"

python train.py --train_path $TRAIN_PATH \
                --max_char_len $MAX_CHAR_LEN  \
                --dev_path $DEV_PATH \
                --max_seq_length $MAX_SEQ_LENGTH \
                --batch_size $BATCH_SIZE \
                --char_embedding_dim $CHAR_EMBEDDING_DIM \
                --char_hidden_dim $CHAR_HIDDEN_DIM \
                --num_layer_bert $NUM_BERT_LAYER \
                --char_vocab_size $CHAR_VOCAB_SIZE \
                --hidden_dropout_prob $HIDDEN_DROPOUT_PROB \
                --hidden_dim $HIDDEN_DIM \
                --num_labels $NUM_LABELS \
                --model_name_or_path $MODEL_NAME_OR_PATH \
                --num_epochs $NUM_EPOCHS \
                --learning_rate $LEARNING_RATE \
                --adam_epsilon $ADAM_EPSILON \
                --weight_decay $WEIGHT_DECAY \
                --warmup_steps $WARMUP_STEPS \
                --max_grad_norm $MAX_GRAD_NORM  \
                --save_folder $SAVE_FOLDER \
                --test_path $TEST_PATH \
                --char_vocab_path $CHAR_VOCAB_PATH \
                --label_set_path $LABEL_SET_PATH \
                --do_eval \
                --do_train \
                --use_char \