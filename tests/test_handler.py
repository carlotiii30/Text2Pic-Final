import json
from unittest.mock import Mock

import pytest

from src.handler import Handler


@pytest.fixture
def socket():
    return Mock()


def test_receive_data(socket):
    socket.recv.return_value = b'{"command": "process_text", "text": "Hello, World!"}'
    handler = Handler(socket)
    data = handler._receive_data()
    assert data == '{"command": "process_text", "text": "Hello, World!"}'


def test_process_request_valid_json(socket):
    data = '{"command": "process_text", "text": "Hello, World!"}'
    handler = Handler(socket)
    request = handler._process_request(data)
    assert request == {"command": "process_text", "text": "Hello, World!"}


def test_execute_command_process_text(socket):
    request = {"command": "process_text", "text": "Hello, World!"}
    handler = Handler(socket)
    response = handler._execute_command(request)
    assert response == {
        "status": "error",
        "message": "Command not implemented",
    }


def test_generate_number_error(socket):
    request = {"command": "generate_number", "text": "invalid"}
    handler = Handler(socket)
    response = handler._generate_number(request["text"])
    assert response["status"] == "error"
    assert "message" in response


def test_execute_command_generate_image(socket):
    request = {"command": "generate_image", "text": "Hello, World!"}
    handler = Handler(socket)
    response = handler._execute_command(request)
    assert response == {
        "status": "error",
        "message": "Command not implemented",
    }


def test_execute_command_unknown_command(socket):
    request = {"command": "unknown_command", "text": "Hello, World!"}
    handler = Handler(socket)
    response = handler._execute_command(request)
    assert response == {
        "status": "error",
        "message": "Unknown command: unknown_command",
    }


def test_send_response(socket):
    response = {"status": "success", "message": "Response message"}
    handler = Handler(socket)
    handler._send_response(response)
    socket.sendall.assert_called_once_with(json.dumps(response).encode())
