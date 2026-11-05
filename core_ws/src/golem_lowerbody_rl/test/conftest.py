"""Share the fixtures of the policy runner tests."""

import pytest


@pytest.fixture
def torch_threads():
    """
    Restore torch's intra-op thread count after the test.

    Loading a TorchScript policy calls torch.set_num_threads(runtime.threads), which holds
    for the whole process; the fixture yields the count before the test and puts it back.
    """
    torch = pytest.importorskip('torch', reason='the H1-2 policies run on torch')
    before = torch.get_num_threads()
    yield before
    torch.set_num_threads(before)
