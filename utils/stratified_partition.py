
def stratified_sample_with_redistribution(df, class_col, frac, seed=42):
    """
    Sample `frac` fraction of `df` (~N_total rows) stratified by `class_col`, 
    but if some classes run out, redistribute the shortfall evenly across
    the classes that still have extra examples.
    """
    # 1. Compute targets
    N_total   = int(len(df) * frac)
    classes   = df[class_col].unique().tolist()
    print(f"Classes: {classes}")
    K         = len(classes)
    per_class = N_total // K

    # 2. Shuffle once for reproducibility, then work with indices
    df_shuf = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    idxs_by_class = {
        c: df_shuf.index[df_shuf[class_col] == c].tolist()
        for c in classes
    }

    # 3. First‐pass: take up to `per_class` from each class
    sampled_idxs = []
    major_classes = []
    for c, idxs in idxs_by_class.items():
        take = min(len(idxs), per_class)
        sampled_idxs.extend(idxs[:take])
        # mark as "major" if there are leftovers
        if len(idxs) > per_class:
            major_classes.append(c)
        # keep only the "unused" indices for extra sampling
        idxs_by_class[c] = idxs[take:]

    n_taken = len(sampled_idxs)
    n_rem   = N_total - n_taken

    # 4. Distribute the remainder evenly among the majors
    if n_rem > 0 and major_classes:
        M           = len(major_classes)
        base_extra  = n_rem // M
        extra_first = n_rem % M

        for i, c in enumerate(major_classes):
            extra = base_extra + (1 if i < extra_first else 0)
            # can't take more than we have left
            extra = min(extra, len(idxs_by_class[c]))
            if extra > 0:
                sampled_idxs.extend(idxs_by_class[c][:extra])

    # 5. Final trim & return
    sampled_idxs = sampled_idxs[:N_total]
    sampled = df_shuf.loc[sampled_idxs].reset_index(drop=True)
    return sampled
