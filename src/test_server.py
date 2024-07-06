from unittest.mock import Mock, patch

import pytest

from src.server import Server


@pytest.fixture
def server():
    return Server("localhost", 1234)


def test_client_handler(server):
    client_socket = Mock()
    handler_mock = Mock()
    with patch("src.server.Handler", return_value=handler_mock):
        server.client_handler(client_socket)
    handler_mock.handle.assert_called_once()


def test_server_instance(server):
    assert isinstance(server, Server)
