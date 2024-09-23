import json
import logging

from src import utils
from src.nums import builders, drawing


# pylint: disable=too-few-public-methods
class Handler:
    def __init__(self, socket):
        self.socket = socket

    def handle(self):
        with self.socket:
            data = self._receive_data()
            request = self._process_request(data)
            response = self._execute_command(request)
            self._send_response(response)

    def _receive_data(self):
        return self.socket.recv(1024).decode()

    def _process_request(self, data):
        try:
            return json.loads(data)

        except json.JSONDecodeError as e:
            response = {
                "status": "error",
                "message": f"Error decoding JSON: {str(e)}",
            }

            logging.error("Error decoding JSON: %s", str(e))

            return response

    def _execute_command(self, request):
        command = request.get("command")
        text = request.get("params").get("text")

        if command == "process_text":
            response = self._process_text(text)

        elif command == "generate_number":
            response = self._generate_number(text)

        elif command == "generate_image":
            response = self._generate_image(text)

        else:
            response = {
                "status": "error",
                "message": f"Unknown command: {command}",
            }

            logging.error("Unknown command: %s", command)

        return response

    def _send_response(self, response):
        self.socket.sendall(json.dumps(response).encode())

    def _process_text(self, text):
        response = {
            "status": "error",
            "message": "Command not implemented",
        }

        logging.error("Command not implemented: %s", text)

        return response

    def _generate_number(self, text):
        try:
            generator, discriminator = builders.build_models()
            cond_gan = builders.build_conditional_gan(generator, discriminator)
            cond_gan = utils.load_model_with_weights(
                cond_gan, "data/models/cgan_nums.weights.h5"
            )

            img = drawing.draw_number(text, cond_gan)
            img = img.tolist()

            response = {
                "status": "success",
                "message": "Image generated successfully",
                "image": img,
            }

            logging.info("Image generated successfully: %s", text)

        except IOError as e:
            response = {
                "status": "error",
                "message": f"Error accessing image file: {str(e)}",
            }

            logging.error("Error accessing image file: %s", str(e))

        except ValueError as e:
            response = {
                "status": "error",
                "message": f"Invalid value: {str(e)}",
            }

            logging.error("Invalid value: %s", str(e))

        return response

    def _generate_image(self, text):
        response = {
            "status": "error",
            "message": "Command not implemented",
        }

        logging.error("Command not implemented: %s", text)

        return response
