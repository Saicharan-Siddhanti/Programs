for _ in range(3):
    with torch.no_grad():
        _ = model(**inputs)
