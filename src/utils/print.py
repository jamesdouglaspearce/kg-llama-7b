def trainable_params(model):
    trainable = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable params: {trainable} || All params: {all_params} || % trainable: {100 * trainable/all_params}")