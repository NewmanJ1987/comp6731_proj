import torch
from implementation import dense_max_margin_loss_gaussian

def test_dmml_gaussian_basic():
    features = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0],
    ])
    logits = torch.randn(4, 2)
    labels = torch.tensor([0, 0, 1, 1])

    classifier = torch.nn.Linear(3, 2, bias=False)
    loss, metrics = dense_max_margin_loss_gaussian(
        features, logits, labels, classifier
    )

    assert isinstance(loss, torch.Tensor)
    assert "ce" in metrics
    assert "mm" in metrics
    assert "var" in metrics


def test_margin_violation_increases_mm_loss():
    features_good = torch.tensor([
        [1.0, 0.0, 0.0],  # close to w0
        [0.0, 1.0, 0.0],  # close to w1
    ])
    labels = torch.tensor([0, 1])
    logits = torch.randn(2, 2)

    classifier = torch.nn.Linear(3, 2, bias=False)
    classifier.weight.data = torch.tensor([
        [1.0, 0.0, 0.0],  # class 0 center
        [0.0, 1.0, 0.0],  # class 1 center
    ])

    # Now a margin violation example
    features_bad = torch.tensor([
        [0.5, 0.5, 0.0],  # ambiguous: similar to both centers
        [0.5, 0.5, 0.0],
    ])

    loss_good, m_good = dense_max_margin_loss_gaussian(
        features_good, logits, labels, classifier, beta=0.5
    )

    loss_bad, m_bad = dense_max_margin_loss_gaussian(
        features_bad, logits, labels, classifier, beta=0.5
    )

    assert m_bad["mm"] > m_good["mm"]

def test_gaussian_similarity_with_broadcasting():
    A = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32)
    B = torch.tensor([[1,0,0],[0,1,0]], dtype=torch.float32)
    assert A.shape == (3,3)
    assert B.shape == (2,3)

    A1 = A.unsqueeze(1)     # [N,1,D]
    B1 = B.unsqueeze(0)     # [1,C,D]

    assert A1.shape == (3,1,3)
    assert B1.shape == (1,2,3)
    # Setup features for a broadcasted subtraction
    assert torch.equal(A1, torch.tensor([
        [[
            1,0,0]
        ],
        [
            [0,1,0]
        ],
        [
            [0,0,1]
        ]
    ], dtype=torch.float32))

    # Setup centers for a broadcast subtraction
    assert torch.equal(B1, torch.tensor([
        [
            [1,0,0],
            [0,1,0]
        ]
    ], dtype=torch.float32))

    diff = A1 - B1
    assert diff.shape == (3,2,3)
    assert torch.equal(diff, torch.tensor([
        [
            [0,0,0],
            [1,-1,0]
        ],
        [
            [-1,1,0],
            [0,0,0]
        ],
        [
            [-1,0,1],
            [0,-1,1]
        ]
    ], dtype=torch.float32))

    diff_squared = diff**2
    assert diff_squared.shape == (3,2,3)
    assert torch.equal(diff_squared, torch.tensor([
        [
            [0,0,0],
            [1,1,0]
        ],
        [
            [1,1,0],
            [0,0,0]
        ],
        [
            [1,0,1],
            [0,1,1]
        ]
    ], dtype=torch.float32))

    # calculate sum of squared differences across feature dimensions
    sum_diff = diff_squared.sum(dim=2)
    assert sum_diff.shape == (3,2)
 
    assert torch.equal(sum_diff, torch.tensor([
        [0,2],
        [2,0],
        [2,2]
    ], dtype=torch.float32))

    torch.exp(-sum_diff / (1 ** 2))
    guass_sim = torch.exp(-sum_diff / (1 ** 2))
    assert guass_sim.shape == (3,2)

    # [Gaussian similarity values between feature 1 and centers 1, 2]
    assert torch.equal(guass_sim, torch.tensor([
        [1.0, 0.1353],
        [0.1353, 1.0],
        [0.1353, 0.1353]
    ], dtype=torch.float32))

    
