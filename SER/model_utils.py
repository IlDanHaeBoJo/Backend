def enable_last_k_blocks(model, last_k: int = 4):
    for p in model.wav2vec2.parameters():
        p.requires_grad = False
    
    layers = model.wav2vec2.encoder.layers
    num_layers = len(layers)
    for i in range(num_layers - last_k, num_layers):
        for p in layers[i].parameters():
            p.requires_grad = True
    
    for name, module in model.named_modules():
        if any(k in name for k in ["classifier", "adversary", "speaker_adversary", "pooler", "stats_projector", "projector"]):
            for p in module.parameters():
                p.requires_grad = True