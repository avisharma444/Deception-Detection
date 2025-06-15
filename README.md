ATTENTION-BASED MODEL (HiS-Attention)

1) run the cell with "add_history_to_jsonl" fucntion and the cell jsut below it to get the preprocessed jsons
2) run the imports cell, just below it and the cell containing the model(starts with the positional encoding class)
3) then run the last 2 cells for the evaluation loop and computing eval metrics

weights link:-https://www.kaggle.com/datasets/harsh99429/attention-tranformer-all-embed

GRAPH-BASED MODEL(LieDetectorGAT)
- Run the first cell for setting up the imports and the preprocessed dataset will be saved to the current working directory
- ⁠The first cell will also intialize the model classes  
- ⁠Run the second cell for starting the training loop , best model gets saved to current working directory
- ⁠Run the third cell for evaluation on test set and to generate a classification report of the model’s performance
- ⁠Run the fourth cell to visualize misclassified samples from test