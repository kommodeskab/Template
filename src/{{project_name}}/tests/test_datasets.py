def test_dummy_dataset(dummy_dataset):
    size = len(dummy_dataset)
    assert size == 16, f"Dataset length should be 16, got {size}"
    for i in range(size):
        data = dummy_dataset[i]
        assert "input" in data
        assert "target" in data
        assert data["input"].shape == (10,), "Input shape should be (10,)"
        assert data["target"].shape == (1,), "Target shape should be (1,)"
