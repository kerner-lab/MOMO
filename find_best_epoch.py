
import json
import sys

model = sys.argv[1]
# model_dir = "/scratch-edge/large-mars-model/models/task_arithmetic/pretraining/" + model
model_dir = "/projects/mlia-active-data/data_ASU/mpurohit/" + model + "/pretraining"

'''' Finding best epoch for all instruments '''

def finding_best_epoch_instrument(epoch_loss_pairs):

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch, val_loss in epoch_loss_pairs:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            best_epoch = epoch
            break

    try:
        return best_epoch
    except UnboundLocalError:
        return None


### CTX
ctx_json_path = model_dir + "/ctx/training_metrics.json"
with open(ctx_json_path, "r") as f:
    metrics_ctx = json.load(f)
ctx_epoch_loss_pairs = list(zip(metrics_ctx["epochs"], metrics_ctx["val_loss"]))
print("CTX", finding_best_epoch_instrument(ctx_epoch_loss_pairs))

### HiRISE
hirise_json_path = model_dir + "/hirise/training_metrics.json"
with open(hirise_json_path, "r") as f:
    metrics_hirise = json.load(f)
hirise_epoch_loss_pairs = list(zip(metrics_hirise["epochs"], metrics_hirise["val_loss"]))
print("HiRISE", finding_best_epoch_instrument(hirise_epoch_loss_pairs))

### THEMIS
themis_json_path = model_dir + "/themis/training_metrics.json"
with open(themis_json_path, "r") as f:
    metrics_themis = json.load(f)
themis_epoch_loss_pairs = list(zip(metrics_themis["epochs"], metrics_themis["val_loss"]))
print("THEMIS", finding_best_epoch_instrument(themis_epoch_loss_pairs))

if model != "vit-l-16":
    ### HiRISE_CTX
    themis_json_path = model_dir + "/hirise_ctx/training_metrics.json"
    with open(themis_json_path, "r") as f:
        metrics_themis = json.load(f)
    temp_epoch_loss_pairs = list(zip(metrics_themis["epochs"], metrics_themis["val_loss"]))
    print("HiRISE_CTX", finding_best_epoch_instrument(temp_epoch_loss_pairs))

    ### HiRISE_THEMIS
    themis_json_path = model_dir + "/hirise_themis/training_metrics.json"
    with open(themis_json_path, "r") as f:
        metrics_themis = json.load(f)
    temp_epoch_loss_pairs = list(zip(metrics_themis["epochs"], metrics_themis["val_loss"]))
    print("HiRISE_THEMIS", finding_best_epoch_instrument(temp_epoch_loss_pairs))

    ### THEMIS_CTX
    themis_json_path = model_dir + "/themis_ctx/training_metrics.json"
    with open(themis_json_path, "r") as f:
        metrics_themis = json.load(f)
    temp_epoch_loss_pairs = list(zip(metrics_themis["epochs"], metrics_themis["val_loss"]))
    print("THEMIS_CTX", finding_best_epoch_instrument(temp_epoch_loss_pairs))

    ### HiRISE_CTX_THEMIS
    themis_json_path = model_dir + "/hirise_ctx_themis/training_metrics.json"
    with open(themis_json_path, "r") as f:
        metrics_themis = json.load(f)
    temp_epoch_loss_pairs = list(zip(metrics_themis["epochs"], metrics_themis["val_loss"]))
    print("HiRISE_CTX_THEMIS", finding_best_epoch_instrument(temp_epoch_loss_pairs))



''' Finding epoch where val_loss is equal for all 3 instruments '''

def finding_equal_loss_for_3_instruments(first_epoch_loss_pairs, second_epoch_loss_pairs, third_epoch_loss_pairs):

    matching_epochs = [
        (epoch_a, epoch_b, epoch_c)
        for epoch_a, val_a in first_epoch_loss_pairs
        for epoch_b, val_b in second_epoch_loss_pairs
        for epoch_c, val_c in third_epoch_loss_pairs
        if round(val_a, 2) == round(val_b, 2) == round(val_c, 2)
    ]
    return matching_epochs

def finding_equal_loss_for_2_instruments(first_epoch_loss_pairs, second_epoch_loss_pairs):

    matching_epochs = [
        (epoch_a, epoch_b)
        for epoch_a, val_a in first_epoch_loss_pairs
        for epoch_b, val_b in second_epoch_loss_pairs
        if round(val_a, 2) == round(val_b, 2)
    ]
    return matching_epochs


print("Matching epochs with equal val_loss across all dictionaries (C, H, T):", finding_equal_loss_for_3_instruments(ctx_epoch_loss_pairs, hirise_epoch_loss_pairs, themis_epoch_loss_pairs))
# print("Matching epochs with equal val_loss across all dictionaries (C, H):", finding_equal_loss_for_2_instruments(ctx_epoch_loss_pairs, hirise_epoch_loss_pairs))
# print("Matching epochs with equal val_loss across all dictionaries (H, T):", finding_equal_loss_for_2_instruments(hirise_epoch_loss_pairs, themis_epoch_loss_pairs))
# print("Matching epochs with equal val_loss across all dictionaries (C, T):", finding_equal_loss_for_2_instruments(ctx_epoch_loss_pairs, themis_epoch_loss_pairs))
